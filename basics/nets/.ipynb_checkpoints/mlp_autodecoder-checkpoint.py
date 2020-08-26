import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    A mlp decoder in AE
    """
    def __init__(self, latent_dim, hidden_dim, pixel_dim):
        super(self.__class__, self).__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(2*hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(2*hidden_dim), pixel_dim),
            nn.Sigmoid())
        
        
    def forward(self, latents, images, EPS=1e-9):
        recon = self.hidden(latents)
        B, C, P, _ = images.shape
        images = images.squeeze(1).view(B, P*P)
        ll = - self.binary_cross_entropy(recon, images)
        return recon, ll
    
    def binary_cross_entropy(self, x_mean, x, EPS=1e-9):
        return - (torch.log(x_mean + EPS) * x + 
                  torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)
    