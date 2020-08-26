import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
import numpy as np


def plot_samples(recons, fs=10, data_name=None):
    recons = recons.squeeze().cpu().detach()
    test_batch_size = len(recons)
    gs = gridspec.GridSpec(int(test_batch_size/10), 10)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(fs, fs*int(test_batch_size/10)/ 10))
    for i in range(test_batch_size):
        ax = fig.add_subplot(gs[int(i/10), i%10])
        try:
            ax.imshow(recons[i], cmap='gray', vmin=0, vmax=1.0)
        except:
            ax.imshow(np.transpose(recons[i], (1,2,0)), vmin=0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
    if data_name is not None:
        plt.savefig(data_name + '_samples.png', dpi=300)
        
def compress_tsne(test_data, enc, dec, device, sample_size):
    zs = []
    ys = []
    print('run test set..')
    for (images, labels) in test_data:
        images = images.squeeze().view(images.shape[0], 784).repeat(sample_size, 1, 1).cuda().to(device)
        latents, _ = enc(images)
        latents = latents.squeeze().cpu().detach().numpy()
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