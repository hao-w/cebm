import os
import math
import torch
import torch.nn as nn
import numpy as np
                 
def cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act, batchnorm, **kwargs):
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
        if (i < (len(channels)-1)) or last_act:#Last layer will be customized 
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(act)
            if 'dropout_prob' in kwargs:
                layers.append(nn.Dropout2d(kwargs['dropout_prob']))
            if 'maxpool_kernels' in kwargs and 'maxpool_strides' in kwargs:
                layers.append(nn.MaxPool2d(kernel_size=kwargs['maxpool_kernels'][i], stride=kwargs['maxpool_strides'][i]))
        in_c = out_c
    return nn.Sequential(*layers)

def deconv_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act, batchnorm, **kwargs):
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
        if (i < (len(channels)-1)) or last_act:
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_c)) 
            layers.append(act)
        in_c = out_c
    return nn.Sequential(*layers)

def mlp_block(input_dim, hidden_dims, activation, **kwargs):
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
    for i, out_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(act)
        in_dim = out_dim
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
    
    
def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)

    h = (h_w[0] + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1

    return h, w

def deconv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of deconvolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)
    h = (h_w[0] - 1) * stride[0] - 2 * padding[0]  + (dilation * (kernel_size[0] - 1)) + 1
    w = (h_w[1] - 1) * stride[1] - 2 * padding[1]  + (dilation * (kernel_size[1] - 1)) + 1

    return h, w

def cnn_output_shape(h, w, kernels, strides, paddings):
    h_w = (h, w)
    for i, kernel in enumerate(kernels):
        h_w = conv_output_shape(h_w, kernels[i], strides[i], paddings[i])
    return h_w

def dcnn_output_shape(h, w, kernels, strides, paddings):
    h_w = (h, w)
    for i, kernel in enumerate(kernels):
        h_w = deconv_output_shape(h_w, kernels[i], strides[i], paddings[i])
    return h_w

def wres_block_params(stride, swap_cnn):
    kernels = [3,3]
    paddings = [1,1]
    if swap_cnn:
        strides = [1, stride]
    else:
        strides = [stride, 1]
    return kernels, strides, paddings
        