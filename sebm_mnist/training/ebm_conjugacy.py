import torch
import time
from sebm_mnist.modules.module_ops import save_modules

def train(objective, optimizer, modules, train_data, num_epochs, sample_size, batch_size, CUDA, DEVICE, SAVE_VERSION):
    """
    training function with KL objective
    """
    data_noise_std, sgld_num_steps, sgld_step_size, sgld_noise_std
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = {'loss' : 0.0}
        for b, (images, _) in enumerate(train_data):
            optimizer.zero_grad()
            images = images.squeeze(1).view(-1, 28*28).repeat(sample_size, 1, 1)
            if CUDA:
                images = images.cuda().to(DEVICE)
            
            loss = objective(modules=modules, data_images=images)
            loss.backward()
            optimizer.step()
            metrics['loss'] += loss.detach()
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
    

    
if __name__ == "__main__":
    import os
    import torch
    from sebm_mnist.modules.module_ops import init_modules
    from sebm_mnist.data import load_data
    from sebm_mnist.objectives import sgld
    
    CUDA = torch.cuda.is_available()
    if CUDA:
        DEVICE = torch.device('cuda:0')
    print('torch:', torch.__version__, 'CUDA:', CUDA)
    # optimization hyper-parameters
    num_epochs = 100
    sample_size = 10
    batch_size = 100
    lr = 1e-4
    ## model hyper-parameters
    D = 2 # data point dimensions
    hidden_dim = 400
    pixels_dim = 28*28
    latents_dim = 10
    reparameterized = False
    optimize_priors = False
    ## EBM hyper-parameters
    data_noise_std = 0.1
    sgld_num_steps = 20
    sgld_step_size = 1
    sgld_init_sample_std = 0.1
    sgld_noise_std = 0.01
    SAVE_VERSION = 'ebm-v1' 
    
    ## data directory
    print('Load MNIST dataset...')
    DATA_DIR = '/home/hao/Research/sebm_data/'
    train_data, test_data = load_data(DATA_DIR)
    
    print('Initialize Modules...')
    modules, optimizer = init_modules(pixels_dim=pixels_dim, 
                                         hidden_dim=hidden_dim, 
                                         latents_dim=latents_dim, 
                                         reparameterized=reparameterized, 
                                         CUDA=CUDA, 
                                         DEVICE=DEVICE, 
                                         optimize_priors=optimize_priors, 
                                         LOAD_VERSION=None, 
                                         LR=lr)
    
    train(objective=kl_data, 
          optimizer=optimizer, 
          modules=modules, 
          train_data=train_data, 
          num_epochs=num_epochs, 
          sample_size=sample_size, 
          batch_size=batch_size, 
          CUDA=CUDA, 
          DEVICE=DEVICE, 
          SAVE_VERSION=SAVE_VERSION)
    
    