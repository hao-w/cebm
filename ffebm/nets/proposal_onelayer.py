import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import math

class Proposal(nn.Module):
    """
    A proposal for the energy based model
    """
    def __init__(self, latent_dim, hidden_dim, pixel_dim, in_channel):
        super(self.__class__, self).__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(2*hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(2*hidden_dim), pixel_dim*in_channel),
            nn.Sigmoid())
        self.in_channel = in_channel
        self.pixel_dim = pixel_dim
        
    def forward(self, latents):
        """
        latents of size S * B * P * P * latent_dim
        images of size S * B * P * P * (in_channel*patch_dim2)
        ll of size S * B * P * P 
        """
        images_mean = self.hidden(latents)
        images_dist = Bernoulli(images_mean)
        images = images_dist.sample()
        ll = images_dist.log_prob(images).sum(-1)
        S, B, P, _, _ = images.shape
        return images.view(S, B, P, P, self.in_channel, self.pixel_dim), ll 
            