import os
import torch
from torchvision import datasets, transforms

def load_mnist(DATA_DIR, batch_size, resize=None):
    """
    load MNIST dataset
    """
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
  
    if resize is not None:
        transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))]) 
    else:
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])         

    train_data = torch.utils.data.DataLoader(
                    datasets.MNIST(DATA_DIR, train=True, download=True,
                                   transform=transform),
                    batch_size=batch_size, shuffle=True) 

    test_data = torch.utils.data.DataLoader(
                    datasets.MNIST(DATA_DIR, train=False, download=True,
                                   transform=transform),
                    batch_size=batch_size, shuffle=True) 
    return train_data, test_data