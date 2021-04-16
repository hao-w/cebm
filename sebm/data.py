import os
import torch
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from PIL import Image
import numpy as np

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
        

def load_mnist_heldon(dataset, data_dir, batch_size, heldon_size, train=True, normalize=False, resize=True):
    """
    load dataset
    """
#     print('Note: downsampling function is %s' % transforms.Resize(1))
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    if dataset == 'mnist':
        img_dims = (1, 28, 28)
        if normalize:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,),(0.5,))]) 
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        fullset = datasets.MNIST(data_dir, train=train, download=True,
                                       transform=transform) 
        inds = torch.randperm(len(fullset.data))[:heldon_size]
        fullset.data = fullset.data[inds]
        fullset.targets = fullset.targets[inds]  
        
    elif dataset == 'fashionmnist':
        img_dims = (1, 28, 28)
        if normalize:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,),(0.5,))]) 
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        fullset = datasets.FashionMNIST(data_dir, train=train, download=True,
                                       transform=transform) 
        inds = torch.randperm(len(fullset.data))[:heldon_size]
        fullset.data = fullset.data[inds]
        fullset.targets = fullset.targets[inds]
        
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

        fullset = datasets.CIFAR10(data_dir+'CIFAR10/', train=train, download=True,
                                       transform=transform)
        inds = torch.randperm(len(fullset.data))[:heldon_size]
        fullset.data = fullset.data[inds]
        fullset.targets = np.array(fullset.targets)
        fullset.targets = fullset.targets[inds]
        
        
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
            fullset = datasets.SVHN(data_dir+'SVHN/', split='train', download=True,
                                           transform=transform)
        else:
            fullset = datasets.SVHN(data_dir+'SVHN/', split='test', download=True,
                                           transform=transform)
        inds = torch.randperm(len(fullset.data))[:heldon_size]
        fullset.data = fullset.data[inds]
        fullset.labels = torch.Tensor(fullset.labels).long()
        fullset.labels = fullset.labels[inds]
    else:
        raise NotImplementError

    data = torch.utils.data.DataLoader(fullset, batch_size=batch_size, shuffle=True) 
    return data, img_dims

def load_mnist_heldout(data_dir, batch_size, heldout_class, train=True, normalize=True, resize=True):
    """
    load dataset
    """
#     print('Note: downsampling function is %s' % transforms.Resize(1))
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    img_dims = (1, 28, 28)
    if normalize:
        transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,),(0.5,))]) 
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(data_dir, train=train, download=True,
                                       transform=transform)
    inds = dataset.targets != heldout_class
    dataset.targets = dataset.targets[inds]
    dataset.data = dataset.data[inds]
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    return data, img_dims
        
def load_data(dataset, data_dir, batch_size, train=True, normalize=True, resize=True, shuffle=True):
    """
    load dataset
    """
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
                        batch_size=batch_size, shuffle=shuffle) 

    elif dataset == 'fashionmnist':
        img_dims = (1, 28, 28)
        if normalize:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,),(0.5,))]) 
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        data = torch.utils.data.DataLoader(
                        datasets.FashionMNIST(data_dir, train=train, download=True,
                                       transform=transform),
                        batch_size=batch_size, shuffle=shuffle) 
        
    elif dataset == 'emnist':
        img_dims = (1, 28, 28)
        if normalize:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,),(0.5,))]) 
        else:
            transform = transforms.Compose([transforms.ToTensor()])

        data = torch.utils.data.DataLoader(
                        datasets.EMNIST(data_dir, split='digits', train=train, download=True,
                                       transform=transform),
                        batch_size=batch_size, shuffle=shuffle) 
        
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
                        batch_size=batch_size, shuffle=shuffle)
        
    elif dataset == 'texture':
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
                        datasets.ImageFolder(root=data_dir+'Texture/dtd/images/', 
                                    transform=transform),
                        batch_size=batch_size, shuffle=shuffle)
            
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
                        batch_size=batch_size, shuffle=shuffle)    
        
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
                            batch_size=batch_size, shuffle=shuffle)
        else:
            data = torch.utils.data.DataLoader(
                            datasets.CelebA(data_dir, split='test', taraget_type='attr', download=True,
                                           transform=transform),
                            batch_size=batch_size, shuffle=shuffle)            

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
            train_dataset = datasets.SVHN(data_dir+'SVHN/', split='train', download=True, transform=transform)
            extra_dataset = datasets.SVHN(data_dir+'SVHN/', split='extra', download=True, transform=transform)

            data = torch.utils.data.DataLoader(
#                             train_dataset,
                            torch.utils.data.ConcatDataset([train_dataset, extra_dataset]),
                            batch_size=batch_size, shuffle=shuffle)
        else:
            data = torch.utils.data.DataLoader(
                            datasets.SVHN(data_dir+'SVHN/', split='test', download=True,
                                           transform=transform),
                            batch_size=batch_size, shuffle=shuffle)            


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
                            batch_size=batch_size, shuffle=shuffle)
        else:
            data = torch.utils.data.DataLoader(
                            datasets.ImageNet(data_dir+'ImageNet/', split='test', download=True,
                                           transform=transform),
                            batch_size=batch_size, shuffle=shuffle) 
            
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
                        batch_size=batch_size, shuffle=shuffle)
        
    elif dataset == 'constant_rgb':
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
                        Constant(color_mode='rgb', root=data_dir + 'Constant/', transform=transform),
                        batch_size=batch_size, shuffle=shuffle)

    elif dataset == 'constant_grayscale':
        img_dims = (1, 28, 28)
        if normalize:
            transform = transforms.Compose([transforms.Resize((28,28)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,),(0.5,))])
        else:
            transform = transforms.Compose([transforms.Resize((28,28)),
                                            transforms.ToTensor()])  
        data = torch.utils.data.DataLoader(
                        Constant(color_mode='grayscale', root=data_dir + 'Constant/',transform=transform),
                        batch_size=batch_size, shuffle=shuffle)
        
    else:
        raise ValueError
        
        
    return data, img_dims

def load_data_as_array(dataset, data_dir, train, normalize=False, flatten=True, shuffle=False):
    f = torch.nn.Flatten()
    train_data, img_dims = load_data(dataset, data_dir, 1000, train=train, normalize=normalize, shuffle=shuffle)
    try:
        x = train_data.dataset.data
    except:
        x = train_data.dataset.datasets[0].data
    if type(x).__module__ == np.__name__:
        if flatten:
            x = f(torch.Tensor(x)).numpy()
        else:
            if dataset == 'mnist' or dataset =='fashionmnist':
                x = x[:, None, :, :]
            elif dataset == 'svhn':
                x = x
#                 print(x.shape)
            else:
                x = np.transpose(x, (0, 3, 1, 2))
            
    elif type(x).__module__ == torch.__name__:
        if flatten:
            x = f(x).numpy()
        else:
            if dataset == 'mnist' or dataset =='fashionmnist':
                x = x.numpy()[:, None, :, :]
            elif dataset =='svhn':
                x = x.numpy()
                print(x.shape)
            else:
                x = np.transpose(x.numpy(), (0, 3, 1, 2))
    else:
        raise TypeError
    try:
        y = train_data.dataset.targets
    except:
        y = train_data.dataset.labels
    y = np.array(y)
    x = x / 255.0
    if normalize:
        x = (x - 0.5) / 0.5
    return x, y

def load_data_remove_labels(dataset, data_dir, num_shots, train=True, normalize=False, shuffle=False):
    """
    load a dataset and remove some of the labels for semi-supervised learning
    """
    xs, ys = load_data_as_array(dataset, data_dir, train, normalize=False, flatten=False)        
    if num_shots == -1:
        return xs, ys
    else:
        xs_permuted, ys_permuted = [], []
        classes = np.unique(ys)
        for k in range(len(classes)):
            ys_k = ys[(ys == classes[k])]
            xs_k = xs[(ys == classes[k])]
            ind_k = np.random.permutation(np.arange(len(ys_k)))
            ys_k = ys_k[ind_k]
            xs_k = xs_k[ind_k]
            ys_k[num_shots:] = -1
            ys_permuted.append(ys_k)
            xs_permuted.append(xs_k)
        return np.concatenate(xs_permuted, 0), np.concatenate(ys_permuted, 0)
    
    