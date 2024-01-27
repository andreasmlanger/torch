import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

np.random.seed(1)


def generate_random_linear_data(w, b, size=100):
    """
    Generates random linear data.
    w: weight
    b: bias
    size: number of data points
    """
    x = torch.rand(size, 1).unsqueeze(dim=1)
    y = w * x + b
    return x, y


def generate_random_classification_data(size=100, n=3, w=6, dim=2):
    """
    Creates random data of different classes.
    size: number of data points per class
    n: number of classes
    w: max value
    dim: dimension, e.g. 2 or 3
    """
    inputs = np.vstack([np.random.multivariate_normal(
        mean=np.random.uniform(-w, w, size=(dim,)),  # means between -w and w
        cov=(np.eye(dim) + 1) / 2,  # covariance matrix: 1 on diagonal, 0.5 everywhere else
        size=size,
    ) for _ in range(n)])
    targets = np.concatenate([np.full(size, i) for i in range(n)])
    indices = np.random.permutation(inputs.shape[0])  # shuffle indices
    return inputs[indices], targets[indices]  # return shuffled data


def create_dataloaders(size: int, batch_size: int, train_dir: str, validation_dir: str):
    # Train data & dataloader
    train_transforms = transforms.Compose([
        transforms.Resize(size=(size, size), antialias=True),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),  # from 1 to 31
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms, target_transform=None)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # Validation data & dataloader
    validation_transforms = transforms.Compose([
        transforms.Resize(size=(size, size), antialias=True),
        transforms.ToTensor()
    ])

    validation_data = datasets.ImageFolder(root=validation_dir, transform=validation_transforms, target_transform=None)
    validation_dataloader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, train_data.classes
