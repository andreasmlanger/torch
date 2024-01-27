import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import torch
from torchmetrics import ConfusionMatrix
from torchvision import io, transforms
import os
import random


def plot_linear_regression_model(model, x_train, x_test, y_train, y_test):
    plt.figure(figsize=(4, 4))
    plt.scatter(x_train.cpu(), y_train, color=plt.cm.viridis(0.1), label='Train Data')
    plt.scatter(x_test.cpu(), y_test, color=plt.cm.viridis(0.5), label='Test Data')

    # Make predictions
    with torch.inference_mode():
        y_predict = model(x_test)
        plt.scatter(x_test, y_predict.cpu().detach().numpy(), color=plt.cm.viridis(0.9), label='Prediction')

    plt.title(f'Weight: {model.state_dict()["weight"].item():.3f} | Bias: {model.state_dict()["bias"].item():.3f}')
    plt.legend()
    plt.show()


def plot_classification_model(model, x_train, x_test, y_train, y_test):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_decision_boundary(model, x_train, y_train, 'Train Data')
    plt.subplot(1, 2, 2)
    plot_decision_boundary(model, x_test, y_test, 'Test Data')
    plt.show()


def plot_decision_boundary(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, title=''):
    """
    Plots decision boundaries of model predicting on X in comparison to y.
    """
    # Put everything to CPU for NumPy & Matplotlib
    model.to('cpu')
    x, y = x.to('cpu'), y.to('cpu')

    # Setup prediction boundaries and grid
    lim = torch.cat((x[:, 0], x[:, 1]), dim=0).abs().max().ceil()
    xx, yy = np.meshgrid(np.linspace(-lim, lim, 1001), np.linspace(-lim, lim, 1001))

    # Create features
    x_to_predict_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(x_to_predict_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_predict = torch.softmax(y_logits, dim=1).argmax(dim=1)  # multi-class
    else:
        y_predict = torch.sigmoid(y_logits).round()  # binary

    # Reshape predictions and plot
    y_predict = y_predict.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_predict, cmap='viridis', alpha=0.3)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap='viridis')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.title(title)


def plot_prediction_matrix(model: torch.nn.Module, data):
    y_predictions = []
    model.eval()
    with torch.inference_mode():
        for X, y in list(data):
            y_logit = model(X.unsqueeze(dim=0))
            y_predict = y_logit.argmax(dim=1).cpu().item()
            y_predictions.append(y_predict)
    y_predict_tensor = torch.tensor(y_predictions)

    conf_mat = ConfusionMatrix(num_classes=len(data.classes), task='multiclass')
    conf_mat_tensor = conf_mat(preds=y_predict_tensor, target=data.targets)

    plot_confusion_matrix(
        conf_mat=conf_mat_tensor.numpy(),
        class_names=data.classes,
        figsize=(8, 6)
    )

    plt.tight_layout()
    plt.show()


def plot_mnist_image_predictions(model: torch.nn.Module, data, only_false=False):
    classes = data.classes
    data = list(data)
    random.shuffle(data)
    model.eval()
    with torch.inference_mode():
        for (X, y) in data:
            y_logits = model(X.unsqueeze(dim=0))
            y_probabilities = torch.softmax(y_logits, dim=1)
            max_value, max_index = torch.max(y_probabilities, dim=1)

            if only_false and max_index == y:
                continue  # skip correct predictions

            plt.figure(figsize=(9, 4))

            plt.subplot(1, 2, 1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(X.squeeze(), cmap='viridis')
            plt.title(f'{classes[max_index]} ({int(100 * max_value)}%)')
            plt.xlabel('Actual Item: {}'.format(classes[y]))

            plt.subplot(1, 2, 2)
            plt.grid(False)
            plt.xticks(range(len(classes)), classes, rotation=45)
            plt.yticks([])
            plt.ylim([0, 1])

            this_plot = plt.bar(range(len(classes)), y_probabilities.squeeze(), color='darkslategray')
            this_plot[max_index].set_color('crimson')
            this_plot[y].set_color('steelblue')

            plt.tight_layout()
            plt.show()


def show_image_predictions(model: torch.nn.Module, image_path: str, size: int, classes: list[str], device: str):
    transform = transforms.Compose([transforms.Resize(size=(size, size), antialias=True)])
    while True:
        random_image = random.choice(os.listdir(image_path))
        random_image_path = os.path.join(image_path, random_image)
        predict_and_plot_image(model, random_image_path, classes, transform, device)


def predict_and_plot_image(model: torch.nn.Module, image_path: str, classes: list[str], transform, device='cpu'):
    image = io.read_image(image_path).type(torch.float32) / 255.
    transformed_image = transform(image)

    model.eval()
    with torch.inference_mode():
        image_prediction = model(transformed_image.unsqueeze(dim=0).to(device))  # add batch dimension

    probabilities = torch.softmax(image_prediction, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1)
    class_name = classes[predicted_label.cpu()].capitalize()

    plt.figure(figsize=(5, 5))
    plt.imshow(transformed_image.permute(1, 2, 0))
    plt.axis(False)
    plt.title(f'Prediction: {class_name} ({probabilities.max().cpu() * 100:.1f}%)')
    plt.tight_layout()
    plt.show()
