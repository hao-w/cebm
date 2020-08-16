import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import math

class Proposal(nn.Module):
    """
    A proposal for the energy based model
    """
    def __init__(self, latent_dim, hidden_dim, pixel_dim2):
        super(self.__class__, self).__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(2*hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(2*hidden_dim), pixel_dim2),
            nn.Sigmoid())
        
#         self.q_mean = nn.Sequential(
#             nn.Linear(int(2*hidden_dim), pixel_dim2),
#             nn.Tanh())
        
#         self.q_log_std = nn.Linear(int(2*hidden_dim), pixel_dim2)
        
    def forward(self, latents):
        """
        latents of size S * B * latent_dim
        images of size S * B * (784)
        ll of size S * B 
        """
#         h = self.hidden(latents)
#         q_mean = self.q_mean(h)
#         q_std = self.q_log_std(h).exp()
        q_mean = self.hidden(latents)
        images_dist = Bernoulli(q_mean)
        images = images_dist.sample()
        ll = images_dist.log_prob(images).sum(-1)
        S, B, pixel_dim2 = images.shape
        pixel_dim = int(math.sqrt(pixel_dim2))
        return images.view(S, B, pixel_dim, pixel_dim).unsqueeze(2), ll 
            