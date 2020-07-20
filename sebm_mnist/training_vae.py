import torch
import time
from quasi_conj.modules.vae_ops import save_modules
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def train(optimizer, modules, train_data, num_epochs, sample_size, batch_size, CUDA, DEVICE, SAVE_VERSION):
    """
    training function with KL objective
    """
    (encoder, decoder) = modules
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = {'elbo' : 0.0, 'loss_theta': 0.0}
        for b, (images, _) in enumerate(train_data):
            optimizer.zero_grad()    
            images = images.squeeze(1).view(-1, 28*28).repeat(sample_size, 1, 1)
            if CUDA:
                images = images.cuda().to(DEVICE)
            q_latents = encoder.forward(images=images)
            p_images = decoder(images=images, latents=q_latents['samples'])
            elbo = (p_images['log_prob'] - q_latents['log_prob']).mean()
            loss = - elbo
            loss.backward()
            optimizer.step()
            
            metrics['elbo'] += elbo.detach()
            metrics['loss_theta'] += - p_images['log_prob'].mean().detach()
        save_modules(modules=modules, SAVE_VERSION=SAVE_VERSION)
        logging(metrics=metrics, filename=SAVE_VERSION, average_normalizer=b+1, epoch=epoch)
        time_end = time.time()
        print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, num_epochs, time_end - time_start))
        
def logging(metrics, filename, average_normalizer, epoch):
    if epoch == 0:
        log_file = open('../results/log-' + filename + '.txt', 'w+')
    else:
        log_file = open('../results/log-' + filename + '.txt', 'a+')
    metrics_print = ",  ".join(['%s=%.5e' % (k, v / average_normalizer) for k, v in metrics.items()])
    print("Epoch=%d, " % (epoch+1) + metrics_print, file=log_file)
    log_file.close()
    
    
def visualize_reconstructions(modules, images, sample_size, batch_size, figure_size, CUDA, DEVICE, save_name=None):
    num_cols = batch_size
    num_rows = 2
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(figure_size, figure_size * num_rows / num_cols))
    (encoder, decoder) = modules

    for i in range(batch_size):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(images.data.numpy()[i], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    images = images.view(-1, 28*28).repeat(sample_size, 1, 1)
    if CUDA:
        images = images.cuda().to(DEVICE)
    q_latents = encoder.forward(images=images)
    p_images = decoder(images=images, latents=q_latents['samples'])
    for j in range(batch_size):
        recons = p_images['image_means'].mean(0).cpu().view(-1, 28, 28)
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(recons.data.numpy()[j], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    if save_name is not None:
        plt.savefig(save_name + '-recon.png', dpi=300)
        
def sample_batch(test_data, data_ptr):
    for b, (images, labels) in enumerate(test_data):
        if b == data_ptr:
            break
    return images.squeeze(1)