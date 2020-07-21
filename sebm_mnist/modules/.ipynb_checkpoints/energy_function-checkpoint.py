import torch
import torch.nn as nn

class Energy_function(nn.Module):
    """
    An energy function that maps an image x to a scalar value which is the energy E(x)
    """
    def __init__(self, pixels_dim, hidden_dim):
        super(self.__class__, self).__init__()
        
        self.energy = nn.Sequential(
            nn.Linear(pixels_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, int(0.5*hidden_dim)),
            nn.LeakyReLU(),
            nn.Linear(int(0.5*hidden_dim), 1))

    def forward(self, images):
        """
        return the energy function E(x)
        """
        return self.energy(images)
    
