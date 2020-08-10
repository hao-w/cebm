import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import math

class Proposal(nn.Module):
    """
    A proposal for the energy based model
    """
    def __init__(self, latent_dim, hidden_dim, pixel_dim):
        super(self.__class__, self).__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pixel_dim),
            nn.Sigmoid())
        
    def forward(self, latents):
        """
        latents of size S * B * P * P * latent_dim
        images of size S * B * P * P * patch_dim2
        ll of size S * B * P * P 
        """
        images_mean = self.hidden(latents)
        images_dist = Bernoulli(images_mean)
        images = images_dist.sample()
        ll = images_dist.log_prob(images).sum(-1)
        return images, ll 
            