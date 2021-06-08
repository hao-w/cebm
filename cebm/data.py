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
    return a dataloader 
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
    dataset_path = os.path.join(data_dir, data)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        
    if data in ['mnist', 'fmnist', 'emnist', 'constant_grayscale', 'omniglot']:
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
        dataset = datasets.MNIST(dataset_path, train=train, transform=transform, download=True)
    elif data == 'fmnist':
        dataset = datasets.fmnist(dataset_path, train=train, transform=transform, download=True)
    elif data == 'emnist':
        dataset = datasets.EMNIST(dataset_path, split='digits', train=train, transform=transform, download=True)
    elif data == 'cifar10':
        dataset = datasets.CIFAR10(dataset_path, train=train, transform=transform, download=True)
    elif data == 'svhn':
        if train:
            train_part = datasets.SVHN(dataset_path, split='train', transform=transform, download=True)
            extra_part = datasets.SVHN(dataset_path, split='extra', transform=transform, download=True)
            dataset = ConcatDataset([train_part, extra_part])
        else:
            dataset = datasets.SVHN(dataset_path, split='test', transform=transform, download=True)
    elif data == 'celeba':
        dataset = datasets.CelebA(dataset_path, split='train', transform=transform, download=True, target_type='attr')
    elif data == 'texture':
        try:
            dataset = datasets.ImageFolder(root=dataset_path+'/dtd/images/', transform=transform)
        except:
            import requests
            import tarfile
            url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
            r = requests.get(url)
            with open(dataset_path + '/dtd-r1.0.1.tar.gz', 'wb') as f:
                f.write(r.content)
            f.close()
            my_tar = tarfile.open(dataset_path + '/dtd-r1.0.1.tar.gz')
            my_tar.extractall(dataset_path)
            my_tar.close()
            dataset = datasets.ImageFolder(root=dataset_path+'/dtd/images/', transform=transform)
    elif data == 'constant_rgb':
        dataset = Constant(color_mode='rgb', root=dataset_path, transform=transform)
    elif data == 'constant_grayscale':
        dataset = Constant(color_mode='grayscale', root=dataset_path, transform=transform)
    elif data == 'omniglot':
        dataset = Omniglot_Concat(root=dataset_path, train=train, download=True, transform=transform)
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
    
def setup_omniglot_loader(data_dir, way, shot, train=True, normalize=True, split_idx=1200):
    dataset_path = os.path.join(data_dir, 'omniglot')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    img_h, img_w, n_channels = 28, 28, 1
    transform = transforms.Compose([transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    if not normalize:
        del transform.transforms[-1] # should not normalize for VAEs
#     _ = Omniglot_Concat(root=dataset_path, download=True, transform=transform)
    task_loader = Task_Loader(data_dir, way, shot, train, split_idx)   
    return task_loader, img_h, img_w, n_channels

        
class Omniglot_Concat(VisionDataset):
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(Omniglot_Concat, self).__init__(root, transform=transform,
                                       target_transform=target_transform)
        
        self.subsets = ['images_background', 'images_evaluation']
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
            
        self._alphabets = sum([[os.path.join(subset, a) for a in list_dir(os.path.join(self.root, subset))] for subset in self.subsets], [])
        
        self._characters: List[str] = sum([[os.path.join(a, c) for c in list_dir(os.path.join(self.root, a))]
                                           for a in self._alphabets], [])

        self._character_images = [[(image, idx) for image in list_files(os.path.join(self.root, character), '.png')]
                                  for idx, character in enumerate(self._characters)]

#         self._character_images = [[(image, idx) for image in list_files(os.path.join(self.root, self._characters[idx]), '.png')] for idx in split_range]

        self._flat_character_images: List[Tuple[str, int]] = sum(self._character_images, [])
        self.processing()
        
        print('Omniglot full set, characters=%d, images=%d' % \
              (self.__num_classes__(), len(self._flat_character_images)))
            
    def __num_classes__(self) -> int:
        return int(len(self._flat_character_images) / 20)


    def processing(self):
        print('Processing raw data')
        self.processed =  defaultdict(list)
        for idx, character in enumerate(tqdm(self._characters)):
            character_path = os.path.join(self.root, self._characters[idx])
            for image_name in list_files(character_path, '.png'):
                image = Image.open(os.path.join(character_path, image_name), mode='r').convert('L')
                if self.transform:
                    image = self.transform(image)
                self.processed[idx].append(image)
        torch.save(self.processed, os.path.join(self.root, 'processed.pt'))
    
    def __len__(self) -> int:
        return len(self._flat_character_images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self._flat_character_images[index]
        image_path = os.path.join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _check_integrity(self) -> bool:
        for subset in self.subsets:
            if not check_integrity(os.path.join(self.root, subset + '.zip'), self.zips_md5[subset]):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            return    
        for subset in self.subsets:
            zip_filename = subset + '.zip'
            url = self.download_url_prefix + '/' + zip_filename
            download_and_extract_archive(url, self.root, filename=zip_filename, md5=self.zips_md5[subset])

class Task_Loader():
    def __init__(self, data_dir, way, shot, train, split_idx):
        super().__init__()
        self.data_dir = data_dir
        self.way = way
        self.shot = shot
        self.train = train
        self.split_idx = split_idx
        
    def generate_tasks(self):
        """
        return a list of tasks given way-shot
        """
        fullset = torch.load(os.path.join(self.data_dir, 'omniglot', 'processed.pt'))
        if self.train:
            train_episodes = []
            class_groups = torch.randperm(self.split_idx).view(int(self.split_idx / self.way), self.way)
            for g in class_groups:
                task = []
                for c in g:
                    list_instances = fullset[c.item()]
                    random.shuffle(list_instances)
                    task += list_instances[:self.shot]
                random.shuffle(task)
                task = torch.stack(task, 0)
                train_episodes.append(task)
            return train_episodes
        else:
#             raise NotImplementedError('Need to implement task generator for test time')
            test_episodes = []
            test_split_idx = torch.randperm(1623 - self.split_idx) + self.split_idx
            truncate_length = (len(test_split_idx) // self.way) * self.way
            class_groups = test_split_idx[:truncate_length].view(int(truncate_length / self.way), self.way)
            for g in class_groups:
                task_images = []
                task_labels = []
                for c in g:
                    list_instances = fullset[c.item()]
                    random.shuffle(list_instances)
                    task_images += list_instances[:self.shot]
                    task_labels += [torch.ones(self.shot) * c] 
                task_images = torch.stack(task_images, 0)
                task_labels = torch.stack(task_labels, 0)
                test_episodes.append((task_images, task_labels))
            return test_episodes
