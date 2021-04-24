import os
import math
import torch
import torch.nn as nn
import numpy as np
from cebm.utils import cnn_output_shape, dcnn_output_shape

# def cnn_mlp_1out(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, hidden_dim, latent_dim, cnn_last_act=True, mlp_last_act=False, batchnorm=False, dropout=False, **kwargs):
#     """
#     A cnn-mlp-based network with 1 output variable
#     """
#     cnn_block = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, cnn_last_act, batchnorm, dropout, **kwargs)
#     out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
#     cnn_output_dim = out_h * out_w * channels[-1]
#     mlp_block = mlp_block(cnn_output_dim, hidden_dim, latent_dim, activation, mlp_last_act, **kwargs)
#     return nn.Sequential(*(list(cnn_block) + [nn.Flatten()] + list(mlp_block)))
    
# def cnn_mlp_2out(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, hidden_dim, latent_dim, cnn_last_act=True, mlp_last_act=False, batchnorm=False, dropout=False, **kwargs):
#     """
#     A cnn-mlp-based network with 2 output variables
#     """
#     cnn_block = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, cnn_last_act, batchnorm, dropout, **kwargs)
#     out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
#     cnn_output_dim = out_h * out_w * channels[-1]
#     mlp_block1 = mlp_block(cnn_output_dim, hidden_dim, latent_dim, activation, mlp_last_act, **kwargs)
#     mlp_block2 = mlp_block(cnn_output_dim, hidden_dim, latent_dim, activation, mlp_last_act, **kwargs)
#     return nn.Sequential(*(list(cnn_block) + [nn.Flatten()])), mlp_block1, mlp_block2

# def mlp_dcnn(mlp_input_dim, hidden_dim, dcnn_input_height, dcnn_input_width, dcnn_input_channels, channels, kernels, strides, paddings, activation, cnn_last_act=False, mlp_last_act=True, batchnorm=False, dropout=False, **kwargs):
#     """
#     A mlp-cnn-based decoder that reverse the encoding procedure of cnn-based encoder
#     """
#     mlp_output_dim = dcnn_input_height * dcnn_input_width * dcnn_input_channels
#     mlp_block = mlp_block(mlp_input_dim, hidden_dim, mlp_output_dim, activation, mlp_last_act, **kwargs)
#     dcnn_block = _dcnn_block(dcnn_input_height, dcnn_input_width, dcnn_input_channels, channels, kernels, strides, paddings, activation, cnn_last_act, batchnorm, dropout, **kwargs)    
#     return nn.Sequential(*(list(mlp_block) + [Reshape([dcnn_input_channels, dcnn_input_height, dcnn_input_width])] + list(dcnn_block)))
                    
def cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act, batchnorm, dropout, **kwargs):
    """
    building blocks for a convnet.
    each block is in form of:
        Conv2d
        BatchNorm2d(optinal)
        Activation
        Dropout(optional)
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        act = nn.LeakyReLU(negative_slope=kwargs['leak_slope'], inplace=True)
    else:
        act = getattr(nn, activation)()
    assert len(channels) == len(kernels), "length of channels: %s,  length of kernels: %s" % (len(channels), len(kernels))
    assert len(channels) == len(strides), "length of channels: %s,  length of strides: %s" % (len(channels), len(strides))
    assert len(channels) == len(paddings), "length of channels: %s,  length of kernels: %s" % (len(channels), len(paddings))
    layers = []
    in_c = input_channels
    for i, out_c in enumerate(channels):
        layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(act)
        if dropout:
            layers.append(nn.Dropout2d(kwargs['dropout_prob']))
        in_c = out_c
    #Last layer will be customized 
    if not last_act:
        if dropout:
            layers = layers[:-1]
        layers = layers[:-1]
        if batchnorm:
            layers = layers[:-1]
    return nn.Sequential(*layers)

def deconv_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act, batchnorm, dropout, **kwargs):
    """
    building blocks for a deconvnet
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        act = nn.LeakyReLU(negative_slope=kwargs['leak_slope'], inplace=True)
    else:
        act = getattr(nn, activation)()
    assert len(channels) == len(kernels), "length of channels: %s,  length of kernels: %s" % (len(channels), len(kernels))
    assert len(channels) == len(strides), "length of channels: %s,  length of strides: %s" % (len(channels), len(strides))
    assert len(channels) == len(paddings), "length of channels: %s,  length of kernels: %s" % (len(channels), len(paddings))
    layers = []
    in_c = input_channels
    for i, out_c in enumerate(channels):
        layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_c)) 
        layers.append(act)
        in_c = out_c
    if not last_act:
        layers = layers[:-1]
        if batchnorm:
            layers = layers[:-1]
    return nn.Sequential(*layers)

def mlp_block(input_dim, hidden_dim, latent_dim, activation, last_act, **kwargs):
    """
    building blocks for a mlp
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        act = nn.LeakyReLU(negative_slope=kwargs['leak_slope'], inplace=True)
    else:
        act = getattr(nn, activation)()
    layers = []
    in_dim = input_dim
    for i, out_dim in enumerate(hidden_dim):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(act)
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, latent_dim))
    if last_act:
        layers.append(act)
    return nn.Sequential(*layers)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)

class Swish(nn.Module):
    """
    The swish activation function
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)