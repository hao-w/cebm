import torch
import torch.nn as nn

class Energy_function(nn.Module):
    """
    The Vanilla CNN network used in the Anatomy model
    """
    def __init__(self, negative_slope=0.05):
        super(self.__class__, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=4),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=1))
#             nn.LeakyReLU(negative_slope=negative_slope),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=negative_slope),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=negative_slope),
#             nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0))
        
    def forward(self, images):
        """
        return the energy function E(x)
        """
#         B, C, _, _ = images.shape
        h = self.cnn(images)
        return h
    
