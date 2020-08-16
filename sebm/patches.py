import os
import time
import torch
from ffebm.data import load_mnist
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate_patches(mnist_data_dir, patch_size):
    if not os.path.isdir(mnist_data_dir+'MNIST_patch/'):
        os.makedirs(mnist_data_dir+'MNIST_patch/')
    train_path = mnist_data_dir+'MNIST_patch/train_data_%d_by_%d.pt' % (patch_size, patch_size)
    test_path = mnist_data_dir+'MNIST_patch/test_data_%d_by_%d.pt' % (patch_size, patch_size)
    batch_size = 10000
    patches_container = []
    print('Loading MNIST dataset...')
    train_data, test_data = load_mnist(mnist_data_dir, batch_size)

    time_start = time.time()
    print('Generating train data patches of size %d by %d...' % (patch_size , patch_size))
    print('train data contains %d images, so there will be %d patches...' % (len(train_data)*batch_size, len(train_data)*batch_size*(28-patch_size+1)**2))
    for b, (images, _) in enumerate(train_data):
        B, C, P, _ = images.shape
        num_patches = P - patch_size + 1
        for p in range(num_patches):
            for q in range(num_patches):
                patches = images[:, :, p:p+patch_size, q:q+patch_size]
                patches_container.append(patches)
        print('%d / %d' % (b+1, len(train_data)))
    patches_container = torch.cat(patches_container, 0)
    torch.save(patches_container, train_path)
    time_end = time.time()
    print('(%ds) %d patches generated.' % (time_end - time_start, len(patches_container)))

    time_start = time.time()
    patches_container = []
    print('Generating test data patches of size %d by %d...' % (patch_size , patch_size))
    print('test data contains %d images, so there will be %d patches...' % (len(test_data)*batch_size, len(test_data)*batch_size*(28-patch_size+1)**2))
    for b, (images, _) in enumerate(test_data):
        B, C, P, _ = images.shape
        num_patches = P - patch_size + 1
        for p in range(num_patches):
            for q in range(num_patches):
                patches = images[:, :, p:p+patch_size, q:q+patch_size]
                patches_container.append(patches)
        print('%d / %d' % (b+1, len(test_data)))
    patches_container = torch.cat(patches_container, 0)
    torch.save(patches_container, test_path)
    time_end = time.time()
    print('(%ds) %d patches generated.' % (time_end - time_start, len(patches_container)))
    print('Complete!')
    
def load_mnist_patches(patch_data_dir, batch_size, normalizing=None):
    """
    load MNIST patches
    """
    (train_path, test_path) = patch_data_dir
    dataset_mnistpatch_train = PatchDataset(torch.load(train_path))
    dataset_mnistpatch_test = PatchDataset(torch.load(test_path))

    if normalizing is not None:
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((normalizing,), (normalizing,))]) 
    else:
        transform = transforms.Compose([
                    transforms.ToTensor()])    

    train_data = torch.utils.data.DataLoader(dataset_mnistpatch_train,
                                             batch_size=batch_size, 
                                             shuffle=True) 

    test_data = torch.utils.data.DataLoader(dataset_mnistpatch_test,
                                             batch_size=batch_size, 
                                             shuffle=True) 
    return train_data, test_data

class PatchDataset(Dataset):
    """
    pytorch dataset class for patch
    """
    def __init__(self, patches, transform=None):
        self.patches = patches
        self.transform = transform
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.patches[idx])
        else:
            return self.patches[idx]

def vis_patches_one_image(patch_size, mnist_data_dir):
    """
    visualize patches for one random image as demo
    """
    test_batch_size = 1
    _, test_data = load_mnist(mnist_data_dir, test_batch_size)
    for (images, _) in test_data:
        break
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.imshow(images.squeeze(0).squeeze(0), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    B, C, P, _ = images.shape
    num_patches = P-patch_size+1
    gs = gridspec.GridSpec(num_patches, num_patches)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(10, 10))
    for p in range(num_patches):
        for q in range(num_patches):
            patch = images[:, :, p:p+patch_size, q:q+patch_size].squeeze(0).squeeze(0)
            ax = fig.add_subplot(gs[p, q])
            ax.imshow(patch, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            
if __name__ == "__main__":
    patch_size = 3
    mnist_data_dir = '../../sebm_data/'
    # vis_patches_one_image(patch_size, mnist_data_dir)
#     train_path = mnist_data_dir+'MNIST_patch/train_data_%d_by_%d.pt' % (patch_size, patch_size)
#     test_path = mnist_data_dir+'MNIST_patch/test_data_%d_by_%d.pt' % (patch_size, patch_size)
#     patch_data_dir = (train_path, test_path)
    
    generate_patches(mnist_data_dir, patch_size)
    

