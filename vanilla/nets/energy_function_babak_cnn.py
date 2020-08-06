import torch
import torch.nn as nn

class Energy_function(nn.Module):
    """
    A CNN architecture used by Babak
    """
    def __init__(self, negative_slope=0.05):
        super(self.__class__, self).__init__()
        

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True))

        self.fc = nn.Sequential(
            nn.Linear(288, 128),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(128, 1))
        
    def forward(self, images):
        """
        return the energy function E(x)
        """
        B, C, _, _ = images.shape
        h1 = self.cnn(images)
        h2 = self.fc(h1.view(B, 288))    
        return h2
    
