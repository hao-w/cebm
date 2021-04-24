import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from cebm.net import cnn_block, deconv_block, mlp_block
from cebm.utils import cnn_output_shape


class Generator_BIGAN(nn.Module):
    """
    A generator in BIGAN with a Gaussian prior on noise
    """
    def __init__(self, device, channels, kernels, strides, paddings, activation, latent_dim, reparameterized=True, **kwargs):
        super().__init__()
        self.gen_net = deconv_block(1, 1, latent_dim, channels, kernels, strides, paddings, activation, last_act=False, batchnorm=True, dropout=False, **kwargs)
        self.last_act = nn.Tanh()
        self.reparameterized = reparameterized
        self.device = device
        self.latent_dim = latent_dim  
        
    def forward(self, batch_size):
        z = torch.randn((batch_size, self.latent_dim, 1, 1), device=self.device)
        x = self.last_act(self.gen_net(z))
        return z, x
    
class Generator_BIGAN_GMM(Generator_BIGAN):
    def __init__(self, optimize_prior, num_clusters, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, hidden_dim, latent_dim, reparameterized=True, **kwargs):
        super().__init__(device, im_height, im_width, num_clusters, input_channels, channels, kernels, strides, paddings, activation, hidden_dim, latent_dim, reparameterized=True, **kwargs)

        self.prior_means = 0.31 * torch.randn((num_clusters, self.latent_dim), device=device)
        self.prior_log_stds = (5*torch.rand((num_clusters, self.latent_dim), device=device) + 1.0).log()
        if optimize_prior:
            self.prior_means = nn.Parameter(self.prior_means)
            self.prior_log_stds = nn.Parameter(self.prior_log_stds)
        self.prior_pi = torch.ones(num_clusters, device=self.device) / num_clusters
        
    def forward(self, batch_size):
        y = cat(probs=self.prior_pi).sample((batch_size,)).argmax(-1)
        p = Normal(self.prior_means[y], self.prior_log_stds[y].exp())
        z = p.rsample() if self.reparameterized else p.sample()
        x = self.gen_net(z.unsqueeze(-1).unsqueeze(-1))
        return z, x

    
class Discriminator_BIGAN(nn.Module):
    def __init__(self, im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, hidden_dim, latent_dim, **kwargs):
        super().__init__()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=True, dropout=True, leak_slope=0.2, dropout_prob=0.2, **kwargs)
        self.flatten = nn.Flatten()
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
        self.mlp_net = mlp_block(latent_dim+cnn_output_dim, hidden_dim, 1, activation, last_act=False, leak_slope=0.2, **kwargs)
        self.sigmoid = nn.Sigmoid()
        
    def binary_pred(self, x, z):
        h1 = self.flatten(self.conv_net(x))
        return self.sigmoid(self.mlp_net(torch.cat((h1, z.squeeze()), dim=-1))).squeeze(-1)
    
    
class Encoder_BIGAN(nn.Module):
    def __init__(self, im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, hidden_dim, latent_dim, reparameterized=True, **kwargs):
        super().__init__()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=True, dropout=False, leak_slope=0.2, **kwargs)
        self.flatten = nn.Flatten()
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
        self.mean = mlp_block(cnn_output_dim, hidden_dim, latent_dim, activation, last_act=False, leak_slope=0.2, **kwargs)
        self.log_std = mlp_block(cnn_output_dim, hidden_dim, latent_dim, activation, last_act=False, leak_slope=0.2, **kwargs)
        self.reparameterized = reparameterized
        
    def forward(self, x):
        mean, std = self.latent_params(x)
        q = Normal(mean, std)
        z = q.rsample() if self.reparameterized else q.sample()
        return z
    
    def latent_params(self, x):
        h = self.flatten(self.conv_net(x))
        return self.mean(h), self.log_std(h).exp()