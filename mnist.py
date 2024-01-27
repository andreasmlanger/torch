"""
Image prediction of MNIST datasets using PyTorch
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from tqdm.auto import tqdm
from utils.general import acc_fn, print_loss_and_accuracy
from utils.plots import plot_mnist_image_predictions, plot_prediction_matrix

print(torch.__version__, torchvision.__version__)

# Select dataset and neural network
# dataset = 'mnist'  # MNIST
dataset = 'fashion_mnist'  # FashionMNIST

# NN = 'NN'  # normal neural network (~98.0% / ~87.6%)
NN = 'CNN'  # convolutional neural network (~99.2% / ~90.6%)

MODEL_PATH = f'E:/models/{dataset}_{NN}.pth'

BATCH_SIZE = 32
EPOCHS = 10

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
torch.manual_seed(1)

dataset = datasets.MNIST if dataset == 'mnist' else datasets.FashionMNIST

train_data = dataset(
    root='E:/datasets',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

test_data = dataset(
    root='E:/datasets',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

# Create model architecture
if NN == 'CNN':
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 5 * 5, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=10)
    ).to(device)

else:  # NN
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=784, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=10)
    ).to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  # loss function
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)  # stochastic gradient descent


def fit_model(dataloader, train=False):
    loss, acc = 0, 0
    model.train() if train else model.eval()

    for x, y in tqdm(dataloader, desc='Training') if train else dataloader:
        x, y = x.to(device), y.to(device)
        if train:
            y_logits = model(x)
            batch_loss = loss_fn(y_logits, y)  # calculate loss
            optimizer.zero_grad()  # zero the optimizer
            batch_loss.backward()  # backpropagation
            optimizer.step()  # optimize model parameters
        else:
            with torch.inference_mode():
                y_logits = model(x)
            batch_loss = loss_fn(y_logits, y)  # calculate loss

        y_predict = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss += batch_loss  # accumulate loss
        acc += acc_fn(y, y_predict)  # accumulate accuracy

    return loss / len(dataloader), acc / len(dataloader)  # normalize with dataloader length


try:
    model.load_state_dict(torch.load(MODEL_PATH))  # load model (i.e. state_dict) if it exists
    test_loss, test_acc = fit_model(test_dataloader)
    print_loss_and_accuracy(test_loss=test_loss, test_acc=test_acc)
except (FileNotFoundError, RuntimeError):
    for n in range(EPOCHS):
        train_loss, train_acc = fit_model(train_dataloader, train=True)
        test_loss, test_acc = fit_model(test_dataloader)
        print_loss_and_accuracy(n, EPOCHS,
                                train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc)
    torch.save(model.state_dict(), MODEL_PATH)  # save trained model

plot_prediction_matrix(model, test_data)
plot_mnist_image_predictions(model, test_data)
