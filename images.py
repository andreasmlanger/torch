"""
CNN, residual neural network (ResNet) and vision transformer (VIT) to classify images
Cats & Dogs: https://www.kaggle.com/c/dogs-vs-cats
Rooms: https://towardsdatascience.com/image-classifier-house-room-type-classification-using-monk-library-d633795a42ef
Move images into labeled folders in 'training' and 'validation' directories

Tensorboard:
- conda activate PyCharm
- cd "G:/My Drive/Coding/AI/torch"
- tensorboard --logdir runs
- http://localhost:6006
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.auto import tqdm
import os
from utils.data import create_dataloaders
from utils.general import acc_fn, print_loss_and_accuracy, print_model_summary, create_writer
from utils.plots import show_image_predictions

# dataset = 'cat_dog'  # cat & dog images
dataset = 'rooms'  # images of rooms (bedroom, kitchen, living room, etc.)

# NN = 'CNN'  # convolutional neural network (~85.9%)
NN = 'RNN'  # residual neural network (ResNet) (~99.4%)
# NN = 'VIT'  # vision transformer (~98.4%)

MODEL_PATH = f'E:/models/{dataset}_{NN}.pth'

EPOCHS = 75 if NN == 'CNN' else 5
SIZE = 92 if NN == 'CNN' else 224  # image size in pixels
BATCH_SIZE = 512 if NN == 'VIT' else 32

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

torch.manual_seed(1)

# Directories for training, validation and test data
BASE_DIR = f'E:/images/{dataset}'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# Create dataloaders
train_dataloader, validation_dataloader, classes = create_dataloaders(SIZE, BATCH_SIZE, TRAIN_DIR, VALIDATION_DIR)

# Load model architecture and send to device
if NN == 'RNN':
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False  # freeze pre-trained weights, so they don't get changed during training
    model.fc = nn.Sequential(
        nn.Linear(in_features=model.fc.in_features, out_features=512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, len(classes), bias=True)
    )
elif NN == 'VIT':
    model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False  # freeze pre-trained weights, so they don't get changed during training
    model.heads = nn.Linear(in_features=768, out_features=len(classes))
else:  # CNN
    hidden_units = 10  # works best
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=hidden_units * 20 * 20, out_features=len(classes))
    )
model.to(device)

# Print model summary
print_model_summary(model, BATCH_SIZE, (3, SIZE, SIZE))

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  # loss function
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.001)


def fit_model(dataloader: torch.utils.data.DataLoader, train: bool = False):
    loss, acc = 0, 0
    model.train() if train else model.eval()

    for x, y in tqdm(dataloader, desc=f'Training   {NN}' if train else f'Validation {NN}'):
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
        loss += batch_loss.item() if train else batch_loss  # accumulate loss
        acc += acc_fn(y, y_predict)  # accumulate accuracy

    return loss / len(dataloader), acc / len(dataloader)  # normalize with dataloader length


def train_model(epochs: int):
    writer = create_writer(name=NN)  # tensorboard writer
    for n in range(epochs):
        train_loss, train_acc = fit_model(train_dataloader, train=True)
        val_loss, val_acc = fit_model(validation_dataloader)
        print_loss_and_accuracy(n, EPOCHS,
                                train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)

        # Experiment tracking
        writer.add_scalars(main_tag='Loss',
                           tag_scalar_dict={'Train': train_loss, 'Validation': val_loss},
                           global_step=n)
        writer.add_scalars(main_tag='Accuracy',
                           tag_scalar_dict={'Train': train_acc, 'Validation': val_acc},
                           global_step=n)
        writer.add_graph(model=model, input_to_model=torch.randn(BATCH_SIZE, 3, SIZE, SIZE).to(device))

    writer.close()


def test_model():
    test_loss, test_acc = fit_model(validation_dataloader)
    print_loss_and_accuracy(val_loss=test_loss, val_acc=test_acc)


try:
    model.load_state_dict(torch.load(MODEL_PATH))  # load model (i.e. state_dict) if it exists
    # test_model()  # optional: test model on validation data after loading
except (FileNotFoundError, RuntimeError):
    train_model(epochs=EPOCHS)
    torch.save(model.state_dict(), MODEL_PATH)  # save trained model

# Predict unseen images from TEST_DIR
show_image_predictions(model, image_path=TEST_DIR, size=SIZE, classes=classes, device=device)
