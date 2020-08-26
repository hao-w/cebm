import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class Encoder(nn.Module):
    """
    A MLP encoder in VAE
    """
    
    def __init__(self, latent_dim, hidden_dim, pixel_dim, reparameterized=False):
        super(self.__class__, self).__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(pixel_dim, int(2*hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(2*hidden_dim), hidden_dim),
            nn.ReLU())
        
        self.q_mu = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim))
        
        self.q_log_sigma = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim))
        
        self.reparameterized = reparameterized
        
    def forward(self, images):
        h = self.hidden(images)
        q_mu = self.q_mu(h)
        q_sigma = self.q_log_sigma(h).exp()
        q_dist = Normal(q_mu, q_sigma)
        if self.reparameterized:
            latents = q_dist.rsample()
        else:
            latents = q_dist.sample()
        q_log_pdf = q_dist.log_prob(latents).sum(-1)
        return latents, q_log_pdf
            