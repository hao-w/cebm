import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import numpy as np

def TSNE_latents(modules, test_data, sample_size, batch_size, CUDA, DEVICE):
    (sampler_latents, _) = modules
    LATENTS = []
    LABELS = []
    time_start = time.time()
    for b, (images, labels) in enumerate(test_data):
        images = images.squeeze(1).view(-1, 28*28).repeat(sample_size, 1, 1)
        if CUDA:
            images = images.cuda().to(DEVICE)
        q_latents = sampler_latents.sample_from_posterior(images=images)
        LATENTS.append(q_latents['samples'].mean(0).cpu().data.numpy())
        LABELS.append(labels.data.numpy())
    LATENTS = np.concatenate(LATENTS, 0)
    LABELS = np.concatenate(LABELS, 0)
    time_end = time.time()
    print('latent sampling completed in %ds' % (time_end - time_start))
    
    time_start = time.time()
    tsne_embedding = TSNE().fit_transform(LATENTS)
    time_end = time.time()
    print('TSNE transformation completed in %ds' % (time_end - time_start))    
    return tsne_embedding, LABELS

def visualize_tsne_embedding(tsne_embedding, labels, figure_size, save_name=None):
    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = plt.gca()
    colors = []
    for k in range(10):
        m = (labels == k)
        p = ax.scatter(tsne_embedding[m, 0], tsne_embedding[m, 1], label='y=%d' % k, alpha=0.5, s=5)
        colors.append(p.get_facecolor())
    ax.legend()
    fig.tight_layout()
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    if save_name is not None:
        plt.savefig(save_name + '.svg', dpi=300)
        
def sample_batch(test_data, data_ptr):
    for b, (images, labels) in enumerate(test_data):
        if b == data_ptr:
            break
    return images.squeeze(1)

def visualize_reconstructions(modules, images, sample_size, batch_size, figure_size, CUDA, DEVICE, save_name=None):
    num_cols = batch_size
    num_rows = 3
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(figure_size, figure_size * num_rows / num_cols))
    (sampler_latents, sampler_images) = modules
    
    for i in range(batch_size):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(images.data.numpy()[i], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    images = images.view(-1, 28*28).repeat(sample_size, 1, 1)
    if CUDA:
        images = images.cuda().to(DEVICE)
    q_latents = sampler_latents.forward(images, sampler_images.prior_nat1, sampler_images.prior_nat2)
    p_images = sampler_images.forward(images=images, latents=q_latents['samples'])
    for j in range(batch_size):
        recons = p_images['image_means'].mean(0).cpu().view(-1, 28, 28)
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(recons.data.numpy()[j], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    if save_name is not None:
        plt.savefig(save_name + '-recon.png', dpi=300)

def visualize_reconstructions_from_prior(modules, test_data, sample_size, batch_size, figure_size, CUDA, DEVICE):
    num_cols = batch_size
    num_rows = 3
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(figure_size, figure_size * num_rows / num_cols))
    (sampler_latents, sampler_images) = modules
    for b, (images, labels) in enumerate(test_data):
        images = images.squeeze(1)
        for i in range(batch_size):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(images.data.numpy()[i], cmap='Greys')
            ax.set_xticks([])
            ax.set_yticks([])
        images = images.view(-1, 28*28).repeat(sample_size, 1, 1)
        if CUDA:
            images = images.cuda().to(DEVICE)
        p_latents = sampler_latents.sample_from_prior(sample_size=sample_size, batch_size=batch_size)
        p_images = sampler_images.forward(latents=p_latents['samples'])
        for j in range(batch_size):
            recons = p_images['image_means'].mean(0).cpu().view(-1, 28, 28)
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(recons.data.numpy()[j], cmap='Greys')
            ax.set_xticks([])
            ax.set_yticks([])
        break
        
       