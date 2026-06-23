import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_cnn(
    conv_channels=(32, 64),
    kernel_size=3,
    pool_size=2,
    fc_hidden=128,
    dropout=0.25,
    activation="relu",
    loss="cross_entropy",
    learning_rate=1e-3,
):
    in_channels=1
    num_classes=10
    input_size=28
    """
    Build a configurable CNN for MNIST-style image classification.

    Args:
        in_channels:   Number of input channels (1 for grayscale MNIST).
        num_classes:   Number of output classes (10 for MNIST digits).
        conv_channels: Tuple of output channels for each conv block.
        kernel_size:   Convolution kernel size.
        pool_size:     Max-pool window size (and stride).
        fc_hidden:     Number of units in the hidden fully-connected layer.
        dropout:       Dropout probability before the final layer.
        activation:    'relu', 'leaky_relu', 'gelu', or 'tanh'.
        loss:          'cross_entropy', 'nll', 'multi_margin', or 'mse'.
        input_size:    Height/width of the (square) input image.
        learning_rate: Learning rate for the optimizer.

    Returns:
        (model, criterion, optimizer): the nn.Module CNN, the chosen loss function, and the optimizer.
    """
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    if activation not in activations:
        raise ValueError(f"activation must be one of {list(activations)}")
    act_fn = activations[activation]

    losses = {
        "cross_entropy": nn.CrossEntropyLoss,
        "nll": nn.NLLLoss,
        "multi_margin": nn.MultiMarginLoss,
        "mse": nn.MSELoss,
    }
    if loss not in losses:
        raise ValueError(f"loss must be one of {list(losses)}")
    criterion = losses[loss]()

    layers = []
    channels = in_channels
    spatial = input_size
    pad = kernel_size // 2  # 'same' padding keeps spatial size before pooling

    for out_ch in conv_channels:
        layers.append(nn.Conv2d(channels, out_ch, kernel_size, padding=pad))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(act_fn())
        layers.append(nn.MaxPool2d(pool_size))
        channels = out_ch
        spatial = spatial // pool_size

    flat_features = channels * spatial * spatial

    layers.append(nn.Flatten())
    layers.append(nn.Linear(flat_features, fc_hidden))
    layers.append(act_fn())
    layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(fc_hidden, num_classes))

    # NLLLoss expects log-probabilities, so append LogSoftmax in that case.
    if loss == "nll":
        layers.append(nn.LogSoftmax(dim=1))

    model = nn.Sequential(*layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer, device


def get_mnist_data(data_dir="./data", batch_size=64, num_workers=2):
    """Download (if needed) and return MNIST train/test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST stats
    ])

    train_set = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def train(model, loader, optimizer, criterion, device, num_classes=10):
    model.train()
    total_loss = 0.0
    use_mse = isinstance(criterion, nn.MSELoss)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        if use_mse:  # MSE needs one-hot float targets, not class indices
            targets = torch.nn.functional.one_hot(
                labels, num_classes
            ).float()
            loss = criterion(logits, targets)
        else:
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)
