import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

class Decoder(nn.Module):
    """
    A decoder in a VAE
    """
    def __init__(self, latents_dim, hidden_dim, pixels_dim, CUDA, DEVICE):
        super(self.__class__, self).__init__()
        self.digit_mean = nn.Sequential(nn.Linear(latents_dim, int(0.5*hidden_dim)),
                                    nn.ReLU(),
                                    nn.Linear(int(0.5*hidden_dim), hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, pixels_dim),
                                    nn.Sigmoid())
        
        self.prior_mean = torch.zeros(latents_dim)
        self.prior_std = torch.ones(latents_dim)  
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.prior_mean = self.prior_mean.cuda()
                self.prior_std = self.prior_std.cuda()   
                
    def forward(self, images, latents):
        p_images = dict()
        prior_dist = Normal(loc=self.prior_mean, scale=self.prior_std)
        log_prior = prior_dist.log_prob(latents).sum(-1)
    
        p_images['image_means'] = self.digit_mean(latents)
        bce_loss = self.binary_cross_entropy(p_images['image_means'], images)
        p_images['log_prob'] = log_prior - bce_loss
        return p_images
    
    def binary_cross_entropy(self, x_mean, x, EPS=1e-9):
        return - (torch.log(x_mean + EPS) * x + 
                  torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)
        
# class Decoder(nn.Module):
#     """
#     A decoder in a VAE
#     """
#     def __init__(self, latents_dim, hidden_dim, pixels_dim, CUDA, DEVICE):
#         super(self.__class__, self).__init__()
#         self.reconstruct_images = nn.Sequential(
#                 nn.Linear(latents_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, pixels_dim),
#                 nn.Sigmoid())
        
#         self.prior_mean = torch.zeros(latents_dim)
#         self.prior_std = (torch.ones(latents_dim) * 1.0).sqrt()        
#         if CUDA:
#             with torch.cuda.device(DEVICE):
#                 self.prior_mean = self.prior_mean.cuda()
#                 self.prior_std = self.prior_std.cuda()   
                
#     def forward(self, images, latents):
#         p_images = dict()
#         prior_dist = Normal(loc=self.prior_mean, scale=self.prior_std)
#         log_prior = prior_dist.log_prob(latents).sum(-1)
    
#         p_images['image_means'] = self.reconstruct_images(latents)
#         bce_loss = self.binary_cross_entropy(p_images['image_means'], images)
#         p_images['log_prob'] = log_prior - bce_loss
#         return p_images