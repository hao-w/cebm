import torch
import torch.nn as nn
import numpy as np
from cebm.net import Swish, Reshape, cnn_block, deconv_block, mlp_block
from cebm.utils import cnn_output_shape
import torch.distributions as dists

class Decoder(nn.Module):
    def __init__(self, im_height, im_width, input_channels, channels, kernels, strides, enc_paddings, dec_paddings, activation, hidden_dim, latent_dim, **kwargs):
        super().__init__()
        deconv_input_h, deconv_input_w = cnn_output_shape(im_height, im_width, kernels, strides, enc_paddings)
        hidden_dim.reverse()
        channels.reverse()
        kernels.reverse()
        strides.reverse()
        deconv_input_channels = channels[0]
        channels = channels[1:] + [input_channels]
        mlp_output_dim = deconv_input_channels * deconv_input_h * deconv_input_w
        self.mlp = mlp_block(latent_dim, hidden_dim, mlp_output_dim, activation, last_act=True, **kwargs)
        self.reshape = Reshape([deconv_input_channels, deconv_input_h, deconv_input_w])
        self.deconv_net = deconv_block(deconv_input_h, deconv_input_w, deconv_input_channels, channels, kernels, strides, dec_paddings, activation, last_act=False, batchnorm=False, dropout=False, **kwargs)
        self.deconv_net = nn.Sequential(*(list(self.deconv_net) + [nn.Sigmoid()]))
    def forward(self, z, x):
        pass
    
    def binary_cross_entropy(self, x_mean, x, EPS=1e-9):
        return - (torch.log(x_mean + EPS) * x + 
                  torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1).sum(-1)
    
class Decoder_Gaussian(Decoder):
    def __init__(self, optimize_prior, device, im_height, im_width, input_channels, channels, kernels, strides, enc_paddings, dec_paddings, activation, hidden_dim, latent_dim, **kwargs):
        super().__init__(im_height, im_width, input_channels, channels, kernels, strides, enc_paddings, dec_paddings, activation, hidden_dim, latent_dim, **kwargs)
        self.prior_mean = torch.zeros(latent_dim, device=device)
        self.prior_log_std = torch.zeros(latent_dim, device=device)
        if optimize_prior:
            self.prior_mean = nn.Parameter(self.prior_mean)
            self.prior_log_std = nn.Parameter(self.prior_log_std)
            
    def forward(self, z, x):
        S, B, C, H ,W = x.shape
        h = self.reshape(self.mlp(z.view(S*B, -1)))
        recon = self.deconv_net(h).view(S, B, C, H, W)
        log_pz = dists.Normal(self.prior_mean, self.prior_log_std.exp()).log_prob(z).sum(-1)
        ll = - self.binary_cross_entropy(recon, x)
        return recon, ll, log_pz
    
class Decoder_GMM(Decoder):
    def __init__(self, optimize_prior, device, num_clusters, im_height, im_width, input_channels, channels, kernels, strides, enc_paddings, dec_paddings, activation, hidden_dim, latent_dim, **kwargs):
        super().__init__(im_height, im_width, input_channels, channels, kernels, strides, enc_paddings, dec_paddings, activation, hidden_dim, latent_dim, **kwargs)
        self.prior_means = 0.31 * torch.randn((num_clusters, latent_dim), device=device)
        self.prior_log_stds = (5*torch.rand((num_clusters, latent_dim), device=device) + 1.0).log()
        if optimize_prior:
            self.prior_means = nn.Parameter(self.prior_means)
            self.prior_log_stds = nn.Parameter(self.prior_log_stds)
        self.prior_pi = torch.ones(num_clusters, device=device) / num_clusters

    def forward(self, z, x):
        S, B, C, H ,W = x.shape
        k = dists.OneHotCategorical(probs=self.prior_pi).sample((S*B,)).argmax(-1)
        prior_mean_sampled = self.prior_means[k].view(S,B,-1)
        prior_log_std_sampled = self.prior_log_stds[k].view(S,B,-1)
        recon = self.dec_net(z.view(S*B, -1)).view(S, B, C, H, W)
        log_pz = dists.Normal(prior_mean_sampled, prior_log_std_sampled.exp()).log_prob(z).sum(-1)
        ll = - self.binary_cross_entropy(recon, x)
        return recon, ll, log_pz

class Encoder(nn.Module):
    def __init__(self, reparameterized, im_height, im_width, input_channels, channels, kernels, strides, enc_paddings, activation, hidden_dim, latent_dim, **kwargs):
        super().__init__()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, enc_paddings, activation, last_act=True, batchnorm=False, dropout=False, **kwargs)
        self.flatten = nn.Flatten()
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, enc_paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
        self.mean = mlp_block(cnn_output_dim, hidden_dim, latent_dim, activation, last_act=False, **kwargs)
        self.log_std = mlp_block(cnn_output_dim, hidden_dim, latent_dim, activation, last_act=False, **kwargs)
        self.reparameterized = reparameterized
        
    def forward(self, x):
        S, B, C, H ,W = x.shape
        mean, std = self.latent_params(x)
        q = dists.Normal(mean, std)
        z = q.rsample() if self.reparameterized else q.sample()
        log_qz = q.log_prob(z).sum(-1)
        return z.view(S,B,-1), log_qz.view(S,B)
    
    def latent_params(self, x):
        S, B, C, H ,W = x.shape
        h = self.flatten(self.conv_net(x.view(S*B, C, H, W)))
        return self.mean(h), self.log_std(h).exp()