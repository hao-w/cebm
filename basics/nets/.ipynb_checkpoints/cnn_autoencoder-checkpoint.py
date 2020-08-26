import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    A CNN encoder in AE
    """
    
    def __init__(self, latent_dim):
        super(self.__class__, self).__init__()
        
        self.hidden = nn.Sequential(
            nn.Conv2d(1, latent_dim, kernel_size=3, stride=1))

        
    def forward(self, images):
        B, C_in, P, _ = images.shape
        latents = self.hidden(images).squeeze(-1).squeeze(-1)
        assert latents.shape == (B, 32), "ERROR!"
        return latents