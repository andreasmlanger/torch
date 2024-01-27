"""
Linear regression using PyTorch
"""

import torch
from torch import nn
from utils.data import generate_random_linear_data
from utils.general import print_loss_and_accuracy, split_test_train
from utils.plots import plot_linear_regression_model

print(torch.__version__)

EPOCHS = 2500
MODEL_PATH = 'E:/models/linear_regression.pth'

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
torch.manual_seed(1)

# Generate random linear data
X_data, y_data = generate_random_linear_data(w=0.7, b=0.3)

# Split into training and test data and send to device
X_train, X_test, y_train, y_test = split_test_train(X_data, y_data)
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

# Create model architecture
model = nn.Linear(in_features=1, out_features=1).to(device)

# Loss function and optimizer
loss_fn = nn.L1Loss()  # loss function
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)  # stochastic gradient descent


def fit_model(x, y, train=False):
    if train:
        model.train()
        y_predict = model(x)  # forward pass
        loss = loss_fn(y_predict, y)  # calculate loss
        optimizer.zero_grad()  # zero the optimizer
        loss.backward()  # backpropagation
        optimizer.step()  # optimize model parameter
    else:
        model.eval()
        with torch.inference_mode():
            y_predict = model(x)  # forward pass
        loss = loss_fn(y_predict, y)  # calculate loss
    return loss


try:
    model.load_state_dict(torch.load(MODEL_PATH))  # load model (i.e. state_dict) if it exists
    test_loss = fit_model(X_test, y_test)
    print_loss_and_accuracy(test_loss=test_loss)
except (FileNotFoundError, RuntimeError):
    for n in range(EPOCHS):
        train_loss = fit_model(X_train, y_train, train=True)
        test_loss = fit_model(X_test, y_test)
        print_loss_and_accuracy(n, EPOCHS, train_loss=train_loss, test_loss=test_loss)
    torch.save(model.state_dict(), MODEL_PATH)  # save trained model

plot_linear_regression_model(model, X_train, X_test, y_train, y_test)
