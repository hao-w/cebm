import torch
from torch import Tensor
from typing import Optional, Any
import torch.nn as nn
import numpy as np
from cebm.net import Swish, Reshape, cnn_block, deconv_block, mlp_block
from cebm.utils import cnn_output_shape

class Decoder_Omniglot(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = mlp_block(64, [256,256,256], 256, activation='ELU', last_act=True)
        self.reshape = Reshape([64, 2, 2])
        self.deconv_net = deconv_block(2, 2, 64, [64,64,64,1], [4,3,4,4], [2,2,2,2], [1,1,1,1], 'ReLU', last_act=False, batchnorm=True)
        self.deconv_net = nn.Sequential(*(list(self.deconv_net) + [nn.Sigmoid()]))
        
    def forward(self, z, x):
        pass
    
    def binary_cross_entropy(self, x_mean, x, EPS=1e-9):
        return - (torch.log(x_mean + EPS) * x + 
                  torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1).sum(-1)

class Encoder_Omniglot(nn.Module):
    def __init__(self, im_height, im_width, input_channels, latent_dim):
        super().__init__()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels=[64,64,64,64], kernels=[3,3,3,3], strides=[1,1,1,1], paddings=[1,1,1,1], activation='ReLU', last_act=True, batchnorm=True, maxpool_kernels=[2,2,2,2], maxpool_strides=[2,2,2,2])
        self.flatten = nn.Flatten()
        self.tr_enc = nn.TransformerEncoder(TransformerEncoderLayer(d_model=256, 
                                                                       nhead=4, 
                                                                       dim_feedforward=256, 
                                                                       activation='ELU'),
                                            num_layers=2)
        
        self.mean = nn.Linear(256, 64)
        self.log_std = nn.Linear(256, latent_dim)
        self.reparameterized = True
        
    def forward(self, x):
        mean, std = self.latent_params(x)
        q = dists.Normal(mean, std)
        z = q.rsample() if self.reparameterized else q.sample()
        log_qz = q.log_prob(z).sum(-1)
        return z, log_qz
    
    def latent_params(self, x):
        S, B, C, H ,W = x.shape
        h1 = self.flatten(self.conv_net(x.view(S*B, C, H, W))).view(S, B, -1)
        h2 = self.tr_enc(h1)
        return self.mean(h2), self.log_std(h2).exp()
    
class TransformerEncoderLayer(nn.Module):
    """
    Override the pytorch code by allowing ELU activation and remove the LayerNorm
    
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead=4, dim_feedforward=256, dropout=0.1, activation="elu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = getattr(nn, activation)()

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src