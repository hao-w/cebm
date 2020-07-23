import torch
import torch.nn as nn

class Energy_function(nn.Module):
    """
    An energy function that maps an image x to a scalar value which is the energy E(x)
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0),
#             nn.LeakyReLU(negative_slope=0.01),
# #             nn.AvgPool2d(kernel_size=(2, 2), stride=2),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.01),
# # #             nn.AvgPool2d(kernel_size=(2, 2), stride=2),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
#             nn.LeakyReLU(negative_slope=0.01))
        
        self.cnn1 = nn.Conv2d(1, 64, kernel_size=3, stride=1)
        self.cnn2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0)
        self.cnn3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.cnn4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0)

#         self.fc = nn.Sequential(
#             nn.Linear(1024, 128),
#             nn.LeakyReLU(negative_slope=0.05),
#             nn.Linear(128, 1))

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, images):
        """
        return the energy function E(x)
        """
        B, C, _, _ = images.shape
        h1 = self.cnn1(images)
        h2 = self.cnn2(h1 * torch.sigmoid(h1))
        h3 = self.cnn3(h2 * torch.sigmoid(h2))
        h4 = self.cnn4(h3 * torch.sigmoid(h3))
        h4_flat = h4.view(B, 1024)
        h5 = self.fc1(h4_flat * torch.sigmoid(h4_flat))
        h6 = self.fc2(h5 * torch.sigmoid(h5))
#         h = self.cnn(images)
#         h6 = self.fc(h.view(B, C*2*2*256))
        return h6
    
