import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from cebm.net import cnn_mlp_1out, cnn_mlp_2out, dcnn, mlp


class Discriminator_GAN(nn.Module):
    """
    Discriminator in a GAN
    """
    def __init__(self, **kwargs):
        super().__init__()
        #To prevent from re-defining 'hidden_dim' and 'latent_dim' for encoders with scalar output and vector output, we manually merge the latent_dim into hidden_dim for scalar output, and re-define the latent_dim as 1.
        kwargs['hidden_dim'].append(kwargs['latent_dim'])
        kwargs['latent_dim'] = 1
        self.disc_net = nn.Sequential(*(list(cnn_mlp_1out(**kwargs)) + [nn.Sigmoid()]))
        
    def latent(self, x):
        return self.disc_net[:-3](x)
    
    def binary_pred(self, x):
        return self.disc_net(x).squeeze()
    
class Generator_GAN(nn.Module):
    """
    A generator in GAN with a spherical Gaussian prior
    """
    def __init__(self, optimize_prior, device, reparameterized=False, **kwargs):
        super().__init__()
        self.gen_net = dcnn(**kwargs)
        self.gen_net = nn.Sequential(*(list(self.gen_net) + [nn.Tanh()]))
        self.noise_mean = torch.zeros(kwargs['input_channels']).to(device)
        self.noise_log_std = torch.zeros(kwargs['input_channels']).to(device)
        if optimize_prior:
            self.noise_mean = nn.Parameter(self.noise_mean)
            self.noise_log_std = nn.Parameter(self.noise_log_std)
        self.reparameterized = reparameterized
        
    def forward(self, S):
        p = Normal(self.noise_mean, self.noise_log_std.exp())
        z = p.rsample((S,)) if self.reparameterized else p.sample((S,))
        x = self.gen_net(z.unsqueeze(-1).unsqueeze(-1))
        return z, x

class Generator_GAN_GMM(nn.Module):
    """
    A generator in GAN where noise is sampled from a GMM prior
    """
    def __init__(self, optimize_prior, device, num_clusters, reparameterized=True, **kwargs):
        super(self.__class__, self).__init__()
        self.gen_net = dcnn(**kwargs)
        self.gen_net = nn.Sequential(*(list(self.gen_net) + [nn.Tanh()]))
        self.prior_means = 0.31 * torch.randn((num_clusters, kwargs['input_channels']), device=device)
        self.prior_log_stds = (5*torch.rand((num_clusters, kwargs['input_channels']), device=device) + 1.0).log()
        if optimize_prior:
            self.prior_means = nn.Parameter(self.prior_means)
            self.prior_log_stds = nn.Parameter(self.prior_log_stds)
        self.prior_pi = torch.ones(num_clusters, device=device) / num_clusters
        self.reparameterized = reparameterized
        
    def forward(self, S):
        y = cat(probs=self.prior_pi).sample((S,)).argmax(-1)
        p = Normal(self.prior_means[y], self.prior_log_stds[y].exp())
        z = p.rsample() if self.reparameterized else p.sample()
        x = self.gen_net(z.unsqueeze(-1).unsqueeze(-1))
        return z, x

    
class Discriminator_BIGAN(nn.Module):
    """
    Discriminator in BIGAN
    """
    def __init__(self, **kwargs):
        super().__init__()
        #To prevent from re-defining 'hidden_dim' and 'latent_dim' for encoders with scalar output and vector output, we manually merge the latent_dim into hidden_dim for scalar output, and re-define the latent_dim as 1.
        self.disc_xz_net = mlp(input_dim=int(2*kwargs['latent_dim']), 
                               hidden_dim=[kwargs['latent_dim']], 
                               latent_dim=1, 
                               activation=kwargs['activation'], 
                               last_act=False,
                               leak_slope=kwargs['leak_slope'])
        self.disc_xz_net = nn.Sequential(*(list(self.disc_xz_net) + [nn.Sigmoid()]))
        kwargs['mlp_last_act'] = True
        self.enc_x_net = cnn_mlp_1out(**kwargs)
        
    def binary_pred(self, x, z):
        return self.disc_xz_net(torch.cat((self.enc_x_net(x), z.squeeze()), dim=1)).squeeze()
    

class Encoder_BIGAN(nn.Module):
    """
    An encoder in BIGAN
    """
    def __init__(self, reparameterized, **kwargs):
        super().__init__()
        self.enc_cnn, self.mean_net, self.log_std_net = cnn_mlp_2out(**kwargs)
        self.reparameterized = reparameterized
        
    def forward(self, x):
        mean, std = self.latent_params(x)
        q = Normal(mean, std)
        z = q.rsample() if self.reparameterized else q.sample()
        return z

    def latent_params(self, x):
        h = self.enc_cnn(x)
        return self.mean_net(h), self.log_std_net(h).exp()