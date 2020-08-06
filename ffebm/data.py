import os
import torch
from torchvision import datasets, transforms

def load_mnist(DATA_DIR, batch_size, normalizing=None, resize=None):
    """
    load MNIST dataset
    """
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
  
    if resize is not None:
        if normalizing is not None:
            transform = transforms.Compose([
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize((normalizing,), (normalizing,))]) 
        else:
            transform = transforms.Compose([
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor()])
    else:
        if normalizing is not None:
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((normalizing,), (normalizing,))]) 
        else:
            transform = transforms.Compose([
                    transforms.ToTensor()])    

    train_data = torch.utils.data.DataLoader(
                    datasets.MNIST(DATA_DIR, train=True, download=True,
                                   transform=transform),
                    batch_size=batch_size, shuffle=True) 

    test_data = torch.utils.data.DataLoader(
                    datasets.MNIST(DATA_DIR, train=False, download=True,
                                   transform=transform),
                    batch_size=batch_size, shuffle=True) 
    return train_data, test_data

    