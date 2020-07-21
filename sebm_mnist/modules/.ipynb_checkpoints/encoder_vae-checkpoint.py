import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from sebm_mnist.gaussian_conjugacy import params_to_nats, nats_to_params, posterior_nats, unnormalized_marginal_log_prob

class Encoder(nn.Module):
    """
    A encoder in a VAE
    """
    def __init__(self, pixels_dim, hidden_dim, neural_ss_dim, reparameterized):
        super(self.__class__, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(pixels_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(0.5*hidden_dim)),
            nn.ReLU())
        
        self.q_mean = nn.Linear(int(0.5*hidden_dim), neural_ss_dim)
        self.q_log_std = nn.Linear(int(0.5*hidden_dim), neural_ss_dim)
        
        self.reparameterized = reparameterized
    
    def forward(self, images):
        q_latents = dict()
        hidden = self.hidden(images)
        q_mean = self.q_mean(hidden)
        q_std = self.q_log_std(hidden).exp()
        q_dist = Normal(q_mean, q_std)
        if self.reparameterized:
            latents = q_dist.rsample()
        else:
            latents = q_dist.sample()
        q_latents['samples'] = latents
        q_latents['log_prob'] = q_dist.log_prob(latents).sum(-1)
        return q_latents
            