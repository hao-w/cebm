import math
import torch
import torch.nn as nn
import numpy as np
from cebm.net import Swish, Reshape, cnn_block, deconv_block, mlp_block, cnn_output_shape
import torch.distributions as dists

class Decoder(nn.Module):
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, dec_paddings, **kwargs):
        super().__init__()
        self.device = device
        deconv_input_h, deconv_input_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        channels.reverse()
        kernels.reverse()
        strides.reverse()
        deconv_input_channels = channels[0]
        channels = channels[1:] + [input_channels]
        mlp_output_dim = deconv_input_channels * deconv_input_h * deconv_input_w
        hidden_dims.reverse()
        hidden_dims.append(mlp_output_dim)
        self.mlp = mlp_block(latent_dim, hidden_dims, activation, **kwargs)
        self.reshape = Reshape([deconv_input_channels, deconv_input_h, deconv_input_w])
        self.deconv_net = deconv_block(deconv_input_h, deconv_input_w, deconv_input_channels, channels, kernels, strides, dec_paddings, activation, last_act=False, batchnorm=False, **kwargs)
        self.deconv_net = nn.Sequential(*(list(self.deconv_net) + [nn.Sigmoid()]))
    def forward(self, z, x):
        pass
    
    def binary_cross_entropy(self, x_mean, x, EPS=1e-9):
        return - (torch.log(x_mean + EPS) * x + 
                  torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1).sum(-1)
    
class Decoder_VAE_Gaussian(Decoder):
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, dec_paddings, **kwargs):
        super().__init__(device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, dec_paddings, **kwargs)
        self.prior_mean = torch.zeros(latent_dim, device=self.device)
        self.prior_log_std = torch.zeros(latent_dim, device=self.device)
    
    def forward(self, z, x):
        S, B, C, H ,W = x.shape
        h = self.reshape(self.mlp(z.view(S*B, -1)))
        recon = self.deconv_net(h).view(S, B, C, H, W)
        log_pz = dists.Normal(self.prior_mean, self.prior_log_std.exp()).log_prob(z).sum(-1)
        ll = - self.binary_cross_entropy(recon, x)
        return recon, ll, log_pz
    
class Decoder_VAE_GMM(Decoder):
    def __init__(self, optimize_prior, num_clusters, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, dec_paddings, **kwargs):
        super().__init__(device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, dec_paddings, **kwargs)
        self.prior_means = 0.31 * torch.randn((num_clusters, latent_dim), device=self.device)
        self.prior_log_stds = (5*torch.rand((num_clusters, latent_dim), device=self.device) + 1.0).log()
        if optimize_prior:
            self.prior_means = nn.Parameter(self.prior_means)
            self.prior_log_stds = nn.Parameter(self.prior_log_stds)
        self.prior_pi = torch.ones(num_clusters, device=device) / num_clusters

    def forward(self, z, x):
        S, B, C, H ,W = x.shape
        k = dists.OneHotCategorical(probs=self.prior_pi).sample((S*B,)).argmax(-1)
#         prior_mean_sampled = self.prior_means[k].view(S,B,-1)
#         prior_log_std_sampled = self.prior_log_stds[k].view(S,B,-1)
        recon = self.dec_net(z.view(S*B, -1)).view(S, B, C, H, W)
        p_dist = dists.Normal(self.prior_means[:, None, None, :], 
                              self.prior_log_stds.exp()[:, None, None, :])
        log_pz = p_dist.log_prob(z[None]).sum(-1).logsumexp(dim=0) - math.log(self.prior_means.shape[0])
        ll = - self.binary_cross_entropy(recon, x)
        return recon, ll, log_pz

class Encoder_VAE(nn.Module):
    def __init__(self, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation='ReLU', reparameterized=True, **kwargs):
        super().__init__()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=False, **kwargs)
        self.flatten = nn.Flatten()
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
        self.mlp_net = mlp_block(cnn_output_dim, hidden_dims, activation, **kwargs)
        self.mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_std = nn.Linear(hidden_dims[-1], latent_dim)
        self.reparameterized = reparameterized
        
    def forward(self, x):
        mean, std = self.latent_params(x)
        q = dists.Normal(mean, std)
        z = q.rsample() if self.reparameterized else q.sample()
        log_qz = q.log_prob(z).sum(-1)
        return z, log_qz
    
    def latent_params(self, x):
        S, B, C, H ,W = x.shape
        h = self.mlp_net(self.flatten(self.conv_net(x.view(S*B, C, H, W))).view(S, B, -1))
        return self.mean(h), self.log_std(h).exp()