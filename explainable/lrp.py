"""
Layer-Wise Relevance Propagation (https://iphome.hhi.de/samek/pdf/MonXAI19.pdf)
Works only for VGG16!
YouTube: https://www.youtube.com/watch?v=PDRewtcqmaI
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import copy


def layer_wise_relevance_propagation(model, dataloader, device, classes):
    x, y = next(iter(dataloader))
    x = x.to(device)
    y = y.numpy()
    outputs = model(x).max(1).indices.detach().cpu().numpy()

    for i in range(len(x)):
        image_relevances = apply_lrp_on_vgg16(model, x[i], device)
        image_relevances = image_relevances.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        image_relevances = np.interp(image_relevances, (image_relevances.min(), image_relevances.max()), (0, 1))

        if outputs[i] == y[i]:
            plt.subplot(1, 2, 1)
            plt.imshow(x[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.title(classes[outputs[i]])
            plt.subplot(1, 2, 2)
            plt.imshow(image_relevances[:, :, 0], cmap='seismic')
            plt.axis('off')
            plt.show()
        else:
            print('This image has not been classified correctly')


def apply_lrp_on_vgg16(model, image, device):
    layers = list(model.features) + [model.avgpool] + dense_to_conv(list(model.classifier))
    activations = forward_pass(model, image, layers)
    one_hot_output = generate_one_hot_output(activations[-1], device)
    activations[-1] = one_hot_output
    relevance_score = backward_pass(layers, activations)
    return relevance_score


def dense_to_conv(layers):
    """
    Convert dense layers to convolutional layers for compatibility with LRP.
    """
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            m, n = (512, layer.weight.shape[0]) if i == 0 else (layer.weight.shape[1], layer.weight.shape[0])
            kernel_size = 7 if i == 0 else 1
            new_layer = nn.Conv2d(m, n, kernel_size)
            new_layer.weight = nn.Parameter(layer.weight.view(n, m, kernel_size, kernel_size))
            new_layer.bias = nn.Parameter(layer.bias)
            layers[i] = new_layer
    return layers


def clone_and_transform_layer(layer, g):
    """
    Clone a layer and pass its parameters through the function g.
    """
    layer = copy.deepcopy(layer)
    try:
        layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError:
        pass
    try:
        layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError:
        pass
    return layer


def forward_pass(model, image, layers):
    image = torch.unsqueeze(image, 0)
    activations = [image] + [None] * len(layers)
    for i, layer in enumerate(layers):
        if i == len(model.features) + 1:
            activations[i] = activations[i].reshape((1, 512, 7, 7))
        activation = layer.forward(activations[i])
        if isinstance(layer, nn.modules.pooling.AdaptiveAvgPool2d):
            activation = torch.flatten(activation, start_dim=1)
        activations[i + 1] = activation
    return activations


def generate_one_hot_output(output_activation, device):
    output_activation = output_activation.detach().cpu().numpy()
    max_activation = output_activation.max()
    one_hot_output = np.zeros_like(output_activation)
    one_hot_output[0, np.argmax(output_activation)] = max_activation
    return torch.FloatTensor(one_hot_output).to(device)


def backward_pass(layers, activations):
    relevance_score = activations[-1]
    for i, layer in reversed(list(enumerate(layers))):
        if isinstance(layer, nn.MaxPool2d):
            layer = nn.AvgPool2d(2)
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.Linear):
            activations[i] = activations[i].data.requires_grad_(True)
            rho, incr = get_lrp_parameters(i)
            z = incr(clone_and_transform_layer(layer, rho).forward(activations[i]))  # forward pass
            s = relevance_score / z  # element-wise division
            (z * s.data).sum().backward()  # backward pass
            c = activations[i].grad
            relevance_score = activations[i] * c  # element-wise product
    return relevance_score


def get_lrp_parameters(layer_index):
    def rho(p):
        if layer_index < 17:  # lower layers
            return p + 0.25 * p.clamp(min=0)  # LRP-gamma: favor effect of positive contributions
        return p

    def incr(v):
        if 16 < layer_index < 31:  # middle layers
            return v + 1e-9 + 0.25 * ((v ** 2).mean() ** 0.5).data  # LRP-epsilon: remove some noise
        return v + 1e-9

    return rho, incr
