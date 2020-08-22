import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sebm.data import load_data
import numpy as np

def plot_samples(images_ebm, fs=10, data_name=None):
    test_batch_size = len(images_ebm)
    images_ebm = images_ebm.squeeze().cpu().detach()
    images_ebm = torch.clamp(images_ebm, min=-1, max=1) * 0.5 + 0.5
    gs = gridspec.GridSpec(int(test_batch_size/10), 10)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(fs, fs*int(test_batch_size/10)/ 10))
    for i in range(test_batch_size):
        ax = fig.add_subplot(gs[int(i/10), i%10])
        try:
            ax.imshow(images_ebm[i], cmap='gray', vmin=0, vmax=1.0)
        except:
            ax.imshow(np.transpose(images_ebm[i], (1,2,0)), vmin=0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
    if data_name is not None:
        plt.savefig(data_name + '_samples.png', dpi=300)
        
def compress_tsne(dataset, data_dir, ebm, device, data_noise_std):
    print('Loading dataset=%s...' % dataset)
    test_data, img_dims = load_data(dataset, data_dir, 1000, train=False)
#     if dataset == 'mnist':
#             print('Load MNIST dataset...')
#             im_height, im_width, input_channels = 28, 28, 1
#             train_data, test_data = load_mnist(data_dir, 1000, normalizing=0.5, resize=None)
#     else:
#         raise NotImplementError
    zs = []
    ys = []
    print('run test set..')
    for (images, labels) in test_data:
        images = images.cuda().to(device)
        images = images + data_noise_std * torch.randn_like(images)
        latents = ebm.forward(images).cpu().detach().numpy()
        zs.append(latents)
        ys.append(labels)
    zs = np.concatenate(zs, 0)
    ys = np.concatenate(ys, 0)
    print('transform latent to 2D tsne features..')
    zs2 = TSNE().fit_transform(zs)
    return zs2, ys

def plot_tsne(num_classes, zs2, ys):
    print('plotting tsne figure..')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    colors = []
    for k in range(num_classes):
        m = (ys == k)
        p = ax.scatter(zs2[m, 0], zs2[m, 1], label='y=%d' % k, alpha=0.5, s=5)
        colors.append(p.get_facecolor())
    ax.legend()
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()