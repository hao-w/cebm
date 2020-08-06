import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

class Proposal(nn.Module):
    """
    A proposal for the energy based model
    """
    def __init__(self, latent_dim):
        super(self.__class__, self).__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid())
        
    def forward(self, latents):
        recon = self.hidden(latents)
        recon_dist = Bernoulli(recon)
        recon_samples = recon_dist.sample()
        recon_log_pdf = recon_dist.log_prob(recon_samples).sum(-1)
        return recon_samples, recon_log_pdf
            