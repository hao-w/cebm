import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any
from cebm.net import cnn_block, mlp_block, cnn_output_shape

class CEBM(nn.Module):
    """
    A generic class of CEBM 
    """
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__()
        self.device = device
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=False, **kwargs)
        self.flatten = nn.Flatten()
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
#         self.nss1_net = mlp_block(cnn_output_dim, hidden_dims, latent_dim, activation, last_act=False, **kwargs)
#         self.nss2_net = mlp_block(cnn_output_dim, hidden_dims, latent_dim, activation, last_act=False, **kwargs)
        self.mlp_net = mlp_block(cnn_output_dim, hidden_dims, activation, **kwargs)
        self.nss1_net = nn.Linear(hidden_dims[-1], latent_dim)
        self.nss2_net = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x):
        h = self.mlp_net(self.flatten(self.conv_net(x)))
        nss1 = self.nss1_net(h) 
        nss2 = self.nss2_net(h)
        return nss1, -nss2**2

    def log_partition(self, nat1, nat2):
        """
        compute the log partition of a normal distribution
        """
        return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()  
    
    def nats_to_params(self, nat1, nat2):
        """
        convert a Gaussian natural parameters its distritbuion parameters,
        mu = - 0.5 *  (nat1 / nat2), 
        sigma = (- 0.5 / nat2).sqrt()
        nat1 : natural parameter which correspond to x,
        nat2 : natural parameter which correspond to x^2.      
        """
        mu = - 0.5 * nat1 / nat2
        sigma = (- 0.5 / nat2).sqrt()
        return mu, sigma

    def params_to_nats(self, mu, sigma):
        """
        convert a Gaussian distribution parameters to the natrual parameters
        nat1 = mean / sigma**2, 
        nat2 = - 1 / (2 * sigma**2)
        nat1 : natural parameter which correspond to x,
        nat2 : natural parameter which correspond to x^2.
        """
        nat1 = mu / (sigma**2)
        nat2 = - 0.5 / (sigma**2)
        return nat1, nat2    
    
    def log_factor(self, x, latents, expand_dim=None):
        """
        compute the log factor log p(x | z) for the CEBM
        """
        nss1, nss2 = self.forward(x)
        if expand_dim is not None:
            nss1 = nss1.repeat(expand_dim , 1, 1)
            nss2 = nss2.repeat(expand_dim , 1, 1)
            return (nss1 * latents).sum(2) + (nss2 * (latents**2)).sum(2)
        else:
            return (nss1 * latents).sum(1) + (nss2 * (latents**2)).sum(1) 
    
    def energy(self, x):
        pass
    
    def latent_params(self, x):
        pass
        
    def log_prior(self, latents):
        pass   
    
    
class CEBM_Gaussian(CEBM):
    """
    conjugate EBM with a spherical Gaussian inductive bias
    """
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__(device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs)
        self.ib_mean = torch.zeros(latent_dim, device=self.device)
        self.ib_log_std = torch.zeros(latent_dim, device=self.device)
    
    def energy(self, x):
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_mean, self.ib_log_std.exp())
        logA_prior = self.log_partition(ib_nat1, ib_nat2)
        logA_posterior = self.log_partition(ib_nat1+nss1, ib_nat2+nss2)
        return logA_prior.sum(0) - logA_posterior.sum(1)   
    
    def latent_params(self, x):
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat1 = self.params_to_nats(self.ib_mean, self.ib_log_std.exp()) 
        return self.nats_to_params(ib_nat1+nss1, ib_nat2+nss2) 
    
    def log_prior(self, latents):
        return Normal(self.ib_mean, self.ib_log_std.exp()).log_prob(latents).sum(-1)      
    
class CEBM_GMM(CEBM):
    """
    conjugate EBM with a GMM inductive bias
    """
    def __init__(self, optimize_ib, num_clusters, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__(device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs)
        #Suggested initialization
        self.ib_means = 0.31 * torch.randn((num_clusters, latent_dim), device=self.device)
        self.ib_log_stds = (5*torch.rand((num_clusters, latent_dim), device=self.device) + 1.0).log()
        if optimize_ib:
            self.ib_means = nn.Parameter(self.ib_means)
            self.ib_log_stds = nn.Parameter(self.ib_log_stds)
        self.K = num_clusters
        self.log_K = torch.tensor([self.K], device=self.device).log()
    
    def energy(self, x):
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_means, self.ib_log_stds.exp())
        logA_prior = self.log_partition(ib_nat1, ib_nat2) # K * D
        #FIXME: Currently we only predict the same neural sufficient statistics for all components.
        logA_posterior = self.log_partition(ib_nat1.unsqueeze(0)+nss1.unsqueeze(1), ib_nat2.unsqueeze(0)+nss2.unsqueeze(1)) # B * K * D
        assert logA_prior.shape == (self.K, nss1.shape[1]), 'unexpected shape.'
        assert logA_posterior.shape == (nss1.shape[0], self.K, nss1.shape[-1]), 'unexpected shape.'
        return self.log_K - torch.logsumexp(logA_posterior.sum(2) - logA_prior.sum(1), dim=-1)   
     
    def latent_params(self, x):
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_means, self.ib_log_stds.exp())
        logA_prior = self.log_partition(ib_nat1, ib_nat2) # K * D
        logA_posterior = self.log_partition(ib_nat1.unsqueeze(0)+nss1.unsqueeze(1), ib_nat2.unsqueeze(0)+nss2.unsqueeze(1)) # B * K * D
        probs = torch.nn.functional.softmax(logA_posterior.sum(2) - logA_prior.sum(1), dim=-1)
        means, stds = nats_to_params(ib_nat1.unsqueeze(0)+nss1.unsqueeze(1), ib_nat2.unsqueeze(0)+nss2.unsqueeze(1))
        pred_y_expand = probs.argmax(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, means.shape[2])
        return torch.gather(means, 1, pred_y_expand).squeeze(1), torch.gather(stds, 1, pred_y_expand).squeeze(1)

class IGEBM(nn.Module):
    def __init__(self, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=False, **kwargs)
        self.flatten = nn.Flatten()
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
        hidden_dims.append(latent_dim)
        self.latent_net = mlp_block(cnn_output_dim, hidden_dims, activation, **kwargs)
        self.energy_net = nn.Linear(latent_dim, 1)

    def energy(self, x):
        h = self.flatten(self.conv_net(x))
        return self.energy_net(self.latent_net(h))
    
    def latent(self, x):
        return self.latent_net[:-2](self.flatten(self.conv_net(x)))

####################################################################
class META_CEBM_Omniglot(nn.Module):
    def __init__(self, im_height, im_width, input_channels, channels, kernels, strides, paddings, latent_dim, activation):
        super().__init__()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=True, maxpool_kernels=[2,2,2,2], maxpool_strides=[2,2,2,2])
        self.flatten = nn.Flatten()
        self.tr_enc = nn.TransformerEncoder(TransformerEncoderLayer(d_model=256, 
                                                                    nhead=4, 
                                                                    dim_feedforward=256, 
                                                                    activation='ELU'),
                                            num_layers=2)
        
        self.mean = nn.Linear(256, latent_dim)
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