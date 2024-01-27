import torch
from torchinfo import summary
from datetime import datetime
import os
import time


def split_test_train(x, y, fraction=0.8):
    """
    Splits x and y into training and test data.
    """
    split = int(fraction * len(x))
    return x[:split], x[split:], y[:split], y[split:]


def print_loss_and_accuracy(n=0, epochs=0, **kwargs):
    """
    Prints training loss and accuracy.
    """
    epoch = n + 1
    if epochs < 101 or epoch % max(1, (epochs // 10)) == 0:
        txt = 'Validation' if epochs == 0 else f'Epoch: {epoch:>{len(str(epochs))}}'  # insert whitespace if needed
        for key, value in kwargs.items():
            key = " ".join([v.capitalize() for v in key.split("_")])
            key = key.replace('Acc', 'Accuracy').replace('Val', 'Validation')
            if 'Loss' in key:
                txt += f' | {key}: {value:.3f}'
            elif 'Accuracy' in key:
                txt += f' | {key}: {value:.1f}%'
        print(txt)
        time.sleep(0.05)  # pause for 50 ms to avoid mixing up with tqdm


def acc_fn(y1, y2):
    """
    Accuracy function, returns percentage of correct labels.
    """
    return torch.eq(y1, y2).sum().item() / len(y1.squeeze()) * 100


def print_model_summary(model: torch.nn.Module, batch_size: int, input_shape: tuple[int, int, int]):
    summary(model,
            input_size=(batch_size, *input_shape),
            col_names=['input_size', 'output_size', 'num_params', 'trainable'],
            col_width=20,
            row_settings=['var_names']
            )
    time.sleep(0.05)  # pause for 50 ms to avoid mixing up with tqdm


def create_writer(name: str):
    from torch.utils.tensorboard import SummaryWriter
    timestamp = datetime.now().strftime('%Y-%m-%d')  # current date in YYYY-MM-DD format
    base_dir = 'E:/models/runs'
    log_dir = os.path.join(base_dir, timestamp, name)
    return SummaryWriter(log_dir=log_dir)
