import os
import torch
import random
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from os.path import join
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, list_dir, list_files
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset
    
def setup_data_loader(data, data_dir, num_shots, batch_size, train, normalize, shuffle=True, shot_random_seed=None):
    """
    This method constructs a dataloader for several commonly-used image datasets
    argument:
        data: name of the dataset
        data_dir: path of the dataset
        num_shots: if positive integer, it means the number of labeled examples per class
                   if value is -1, it means the full dataset is labeled
        batch_size: batch size 
        train: if True return the training set;if False, return the test set
        normalize: if True, rescale the pixel values to [-1, 1]
        shot_random_seed: the random seed used when num_shot != -1
    """
        
    if data in ['mnist', 'fashionmnist', 'emnist', 'constant_grayscale']:
        img_h, img_w, n_channels = 28, 28, 1
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
    else:
        img_h, img_w, n_channels = 32, 32, 3
        transform = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))])
    if not normalize:
        del transform.transforms[-1] # should not normalize for VAEs

    if data == 'mnist':
        dataset = datasets.MNIST(data_dir, train=train, transform=transform, download=True)
    elif data == 'fashionmnist':
        dataset = datasets.FashionMNIST(data_dir, train=train, transform=transform, download=True)
    elif data == 'emnist':
        dataset = datasets.EMNIST(data_dir, split='digits', train=train, transform=transform, download=True)
    elif data == 'cifar10':
        data_dir = join(data_dir, 'CIFAR10')
        dataset = datasets.CIFAR10(data_dir, train=train, transform=transform, download=True)
    elif data == 'svhn':
        data_dir = join(data_dir, 'SVHN')
        if train:
            train_part = datasets.SVHN(data_dir, split='train', transform=transform, download=True)
            extra_part = datasets.SVHN(data_dir, split='extra', transform=transform, download=True)
            dataset = ConcatDataset([train_part, extra_part])
        else:
            dataset = datasets.SVHN(data_dir, split='test', transform=transform, download=True)
    elif data == 'texture':
        data_dir = join(data_dir, 'texture')
        try:
            dataset = datasets.ImageFolder(root=dataset_dir+'/dtd/images/', transform=transform)
        except:
            import requests
            import tarfile
            url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
            r = requests.get(url)
            with open(data_dir + '/dtd-r1.0.1.tar.gz', 'wb') as f:
                f.write(r.content)
            f.close()
            my_tar = tarfile.open(data_dir + '/dtd-r1.0.1.tar.gz')
            my_tar.extractall(data_dir)
            my_tar.close()
            dataset = datasets.ImageFolder(root=data_dir+'/dtd/images/', transform=transform)
    elif data == 'constant_rgb':
        data_dir = join(data_dir, 'constant_rgb')
        dataset = Constant(color_mode='rgb', root=data_dir, transform=transform)
    elif data == 'constant_grayscale':
        data_dir = join(data_dir, 'constant_grayscale')
        dataset = Constant(color_mode='grayscale', root=data_dir, transform=transform)
    else:
        raise NotImplementError
    
    if num_shots != -1:
        torch.manual_seed(shot_random_seed)   
        if data == 'svhn':
            if train:
                dataset = dataset.datasets[0]
            labels = dataset.labels
        else:
            labels = dataset.targets
        labels = torch.tensor(labels)
        num_classes = len(torch.unique(labels))
        for k in range(num_classes):
            indk = torch.where(labels == k)[0]
            ind_unlabelled = indk[torch.randperm(len(indk))[num_shots:]]
            labels[ind_unlabelled] = -1
        if data == 'svhn':
            dataset.labels = labels
        else:
            dataset.targets = labels
            
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader, img_h, img_w, n_channels
    
class Constant(VisionDataset):
    def __init__(self, color_mode, root, loader=default_loader, transform=None):
        super(Constant, self).__init__(root, transform=transform)        

        if color_mode == 'rgb':
            self.data_file = 'rgb.pt'
        elif color_mode == 'grayscale':
            self.data_file = 'grayscale.pt'
        else:
            raise ValueError
        self.color_mode = color_mode 
        self.data  = torch.load(os.path.join(self.root, self.data_file))
        
    def __getitem__(self, index):
        img = self.data[index]
        if self.color_mode == 'rgb':
            img = Image.fromarray(img.numpy().astype(np.uint8), mode='RGB')
        elif self.color_mode == 'grayscale':
            img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
        else:
            raise ValueError
            
        if self.transform is not None:
            img = self.transform(img)

        return img, int(-1)

    def __len__(self):
        return len(self.data)