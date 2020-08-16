import os
import torch
import torch.nn as nn
from sebm.util import cnn_output_shape

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
        
def _cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, leak=None):
    """
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        assert isinstance(leak, (float, int)), "type of leak: %s, expected type: float or int." % type(leak)
        act = nn.LeakyReLU(negative_slope=leak, inplace=True)
    else:
        act = getattr(nn, activation)()
    assert len(channels) == len(kernels), "length of channels: %s,  length of kernels: %s" % (len(channels), len(kernels))
    assert len(channels) == len(strides), "length of channels: %s,  length of strides: %s" % (len(channels), len(strides))
    assert len(channels) == len(paddings), "length of channels: %s,  length of kernels: %s" % (len(channels), len(paddings))
    layers = []
    in_c = input_channels
    for i, out_c in enumerate(channels):
        layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
        layers.append(act)
        in_c = out_c
    out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
    flatten_output_dim = out_h * out_w * channels[-1]
    return nn.Sequential(*layers), flatten_output_dim

def _mlp_block(input_dim, hidden_dim, latent_dim, activation, leak=None, last_act=False):
    """
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        assert isinstance(leak, (float, int)), "type of leak: %s, expected type: float or int." % type(leak)
        act = nn.LeakyReLU(negative_slope=leak, inplace=True)
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
    
class SimpleNet(nn.Module):
    """
    """
    def __init__(self, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dim, latent_dim, activation, leak=None):
        super().__init__()
        self.cnn_block, self.mlp_input_dim = _cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, leak=leak)
        self.flatten = nn.Flatten()
        self.mlp_block = _mlp_block(self.mlp_input_dim, hidden_dim, latent_dim, activation, leak=leak)
    def forward(self, x):
        h = self.cnn_block(x)
        return self.mlp_block(self.flatten(h))