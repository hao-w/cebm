import os
import torch
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from PIL import Image
import numpy as np

class Flowers102(VisionDataset):
    data_file = 'data.pt'

    def __init__(self, root, loader=default_loader, transform=None, target_transform=None, download=False):
        super(Flowers102, self).__init__(root, transform=transform,
                                            target_transform=target_transform)        
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, self.data_file))
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray((img.numpy()*255).astype(np.uint8), mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
    
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file))
    def download(self):
        """
        Dolwnload 102Flower data if if doesn't exist in processed_folder already
        """
        if self._check_exists():
            return
        
        import tarfile
        from scipy.io import loadmat
        import numpy as np
        
        try:
            from urllib.request import urlretrieve
        except ImportError:
            from urllib import urlretrieve
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        print('Downloading images from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/ ...')
        image_file = os.path.join(self.raw_folder, "102flowers.tgz")
        urlretrieve("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", image_file)
        # extract flower images from tar file
        print('Extracting ' + image_file + '...')
        tarfile.open(image_file).extractall(path=self.raw_folder)
        # clean up
        os.remove(image_file)
        label_file = os.path.join(self.raw_folder, "imagelabels.mat")
        urlretrieve("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat", label_file)
#         # process and save as torch files
        print('Processing...')
        pre_transforms = transforms.Compose([
                            transforms.Resize((32,32)),
                            transforms.ToTensor()])
        data = []
        for f in range(8189):
            img = Image.open(os.path.join(self.raw_folder, 'jpg', 'image_0%04d.jpg' % (f+1)))
            data.append(pre_transforms(img).unsqueeze(0))
        data = torch.cat(data, 0)
        assert data.shape == (8189, 3, 32, 32)

        targets = torch.Tensor(np.squeeze(loadmat(os.path.join(self.raw_folder, 'imagelabels.mat'))['labels']))
        data_set = (data.permute(0, 2, 3, 1), targets)
        with open(os.path.join(self.processed_folder, self.data_file), 'wb') as f:
            torch.save(data_set, f)
        print('Done.')
        

def load_data(dataset, data_dir, batch_size, train=True, normalize=True, resize=True):
    """
    load dataset
    """
    print('Note: downsampling function is %s' % transforms.Resize(1))
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    if dataset == 'mnist':
        img_dims = (1, 28, 28)
        if normalize:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,),(0.5,))]) 
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        data = torch.utils.data.DataLoader(
                        datasets.MNIST(data_dir, train=train, download=True,
                                       transform=transform),
                        batch_size=batch_size, shuffle=True) 
        
    elif dataset == 'cifar10':
        img_dims = (3, 32, 32)
        if normalize:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,0.5,0.5), 
                                                                 (0.5,0.5,0.5))])
        else:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor()])            

        data = torch.utils.data.DataLoader(
                        datasets.CIFAR10(data_dir+'CIFAR10/', train=train, download=True,
                                       transform=transform),
                        batch_size=batch_size, shuffle=True)
        
    elif dataset == 'cifar100':
        img_dims = (3, 32, 32)
        
        if normalize:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,0.5,0.5), 
                                                                 (0.5,0.5,0.5))])
        else:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor()])  
        data = torch.utils.data.DataLoader(
                        datasets.CIFAR10(data_dir+'CIFAR100/', train=train, download=True,
                                       transform=transform),
                        batch_size=batch_size, shuffle=True)    
        
    elif dataset == 'celeba':
        img_dims = (3, 32, 32)
        if normalize:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,0.5,0.5), 
                                                                 (0.5,0.5,0.5))])
        else:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor()])  
        if train:
            data = torch.utils.data.DataLoader(
                            datasets.CelebA(data_dir, split='train', target_type='attr', download=True,
                                           transform=transform),
                            batch_size=batch_size, shuffle=True)
        else:
            data = torch.utils.data.DataLoader(
                            datasets.CelebA(data_dir, split='test', taraget_type='attr', download=True,
                                           transform=transform),
                            batch_size=batch_size, shuffle=True)            

    elif dataset == 'svhn':
        img_dims = (3, 32, 32)
        if normalize:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,0.5,0.5), 
                                                                 (0.5,0.5,0.5))])
        else:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor()])  
        
        if train:
            data = torch.utils.data.DataLoader(
                            datasets.SVHN(data_dir+'SVHN/', split='train', download=True,
                                           transform=transform),
                            batch_size=batch_size, shuffle=True)
        else:
            data = torch.utils.data.DataLoader(
                            datasets.SVHN(data_dir+'SVHN/', split='test', download=True,
                                           transform=transform),
                            batch_size=batch_size, shuffle=True)            

    elif dataset == 'imagenet':
        img_dims = (3, 32, 32)
        if normalize:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,0.5,0.5), 
                                                                 (0.5,0.5,0.5))])
        else:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor()])  
        
        if train:
            data = torch.utils.data.DataLoader(
                            datasets.ImageNet(data_dir+'ImageNet/', split='train', download=True,
                                           transform=transform),
                            batch_size=batch_size, shuffle=True)
        else:
            data = torch.utils.data.DataLoader(
                            datasets.ImageNet(data_dir+'ImageNet/', split='test', download=True,
                                           transform=transform),
                            batch_size=batch_size, shuffle=True) 
            
    elif dataset == 'flowers102':
        img_dims = (3, 32, 32)
        if normalize:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,0.5,0.5), 
                                                                 (0.5,0.5,0.5))])
        else:
            transform = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor()])  
        data = torch.utils.data.DataLoader(
                        Flowers102(data_dir, download=True,
                                       transform=transform),
                        batch_size=batch_size, shuffle=True)
        
    else:
        raise ValueError
    return data, img_dims
