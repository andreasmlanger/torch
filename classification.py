"""
Classification using PyTorch
"""

import torch
from torch import nn
from utils.data import generate_random_classification_data
from utils.general import acc_fn, print_loss_and_accuracy, split_test_train
from utils.plots import plot_classification_model

print(torch.__version__)

N = 4  # number of classes
EPOCHS = 500
MODEL_PATH = 'E:/models/classification.pth'

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
torch.manual_seed(1)

# Generate random classification data
X_data, y_data = generate_random_classification_data(n=N)

# Convert to PyTorch tensors
X_data = torch.from_numpy(X_data).type(torch.float)
y_data = torch.from_numpy(y_data).type(torch.LongTensor if N > 2 else torch.float)  # 'LongTensor' for CrossEntropy!

# Split into training and test data and send to device
X_train, X_test, y_train, y_test = split_test_train(X_data, y_data)
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

# Create model architecture
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=N if N > 2 else 1)
).to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss() if N > 2 else nn.BCEWithLogitsLoss()  # loss function
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)  # Adam


def fit_model(x, y, train=False):
    if train:
        model.train()
        y_logits = model(x).squeeze()  # forward pass
        loss = loss_fn(y_logits, y)  # calculate loss
        optimizer.zero_grad()  # zero the optimizer
        loss.backward()  # backpropagation
        optimizer.step()  # optimize model parameters
    else:
        model.eval()
        with torch.inference_mode():
            y_logits = model(x).squeeze()  # forward pass
        loss = loss_fn(y_logits, y)

    y_predict = torch.softmax(y_logits, dim=1).argmax(dim=1) if N > 2 else torch.sigmoid(y_logits).round()
    acc = acc_fn(y_predict, y)  # accuracy metric
    return loss, acc


try:
    model.load_state_dict(torch.load(MODEL_PATH))  # load model (i.e. state_dict) if it exists
    test_loss, test_acc = fit_model(X_test, y_test)
    print_loss_and_accuracy(test_loss=test_loss, test_acc=test_acc)
except (FileNotFoundError, RuntimeError):
    for n in range(EPOCHS):
        train_loss, train_acc = fit_model(X_train, y_train, train=True)
        test_loss, test_acc = fit_model(X_test, y_test)
        print_loss_and_accuracy(n, EPOCHS, train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc)
    torch.save(model.state_dict(), MODEL_PATH)  # save trained model

plot_classification_model(model, X_train, X_test, y_train, y_test)
