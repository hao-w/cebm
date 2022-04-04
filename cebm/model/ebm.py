import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any
from cebm.net import cnn_block, deconv_block, mlp_block, cnn_output_shape, Swish, Reshape
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.continuous_bernoulli import ContinuousBernoulli as CB
    
class CEBM(nn.Module):
    """
    A generic class of CEBM 
    """
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__()
        self.device = device
        self.flatten = nn.Flatten()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=False, **kwargs)
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
        self.nss1_net = nn.Linear(cnn_output_dim, latent_dim)
        self.nss2_net = nn.Linear(cnn_output_dim, latent_dim)
            
    def forward(self, x):
        h = self.flatten(self.conv_net(x))
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
            nss1 = nss1.expand(expand_dim , -1, -1)
            nss2 = nss2.expand(expand_dim , -1, -1)
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
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_mean, self.ib_log_std.exp()) 
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
        self.ib_means = torch.randn((num_clusters, latent_dim), device=self.device)
        self.ib_log_stds = torch.ones((num_clusters, latent_dim), device=self.device).log()
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
        means, stds = self.nats_to_params(ib_nat1.unsqueeze(0)+nss1.unsqueeze(1), ib_nat2.unsqueeze(0)+nss2.unsqueeze(1))
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
