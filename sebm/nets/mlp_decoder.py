import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

class Decoder(nn.Module):
    """
    A MLP decoder in VAE
    """
    def __init__(self, latent_dim, hidden_dim, pixel_dim, CUDA, DEVICE):
        super(self.__class__, self).__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(2*hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(2*hidden_dim), pixel_dim),
            nn.Sigmoid())
        
        self.prior_mu = torch.zeros(latent_dim)
        self.prior_sigma = torch.ones(latent_dim)
        
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.prior_mu = self.prior_mu.cuda()
                self.prior_sigma = self.prior_sigma.cuda()
        
    def forward(self, latents, images):
        recon = self.hidden(latents)
        p_log_pdf = Normal(self.prior_mu, self.prior_sigma).log_prob(latents).sum(-1)
#         ll = Bernoulli(recon+EPS).log_prob(images).sum(-1)
        ll = - self.binary_cross_entropy(recon, images)
        return p_log_pdf, recon, ll
            
    def binary_cross_entropy(self, x_mean, x, EPS=1e-9):
        return - (torch.log(x_mean + EPS) * x + 
                  torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)