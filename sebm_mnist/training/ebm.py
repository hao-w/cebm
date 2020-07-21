import torch
import time
from sebm_mnist.objectives import mle

def train(optimizer, ef, sgld_sampler, data_noise_sampler, train_data, num_epochs, sample_size, sgld_num_steps, sgld_step_size, buffer_size, buffer_percent, CUDA, DEVICE, SAVE_VERSION):
    """
    training the energy based model (ebm) by maximizing the marginal likelihood
    """
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = {'loss' : 0.0, 'energy_data' : 0.0, 'energy_ebm' : 0.0}
        for b, (images, _) in enumerate(train_data):
            pixels_size = int(images.shape[-1]*images.shape[-2])
            batch_size = images.shape[0]
            optimizer.zero_grad()
            images = images.squeeze(1).view(-1, pixels_size)
#             images = images.repeat(sample_size, 1, 1)
            if CUDA:
                images = images.cuda().to(DEVICE)
            if data_noise_sampler is not None: ## add Gaussian noise to true data images
                data_noise = data_noise_sampler.sample(sample_size, batch_size, pixels_size).squeeze(0)
                assert images.shape == data_noise.shape, "ERROR! data noise have unexpected shape."
                images = images + data_noise
            energy_data, energy_ebm = mle(ef, sgld_sampler, images, sgld_num_steps, sgld_step_size, buffer_size, buffer_percent)
            loss = energy_data - energy_ebm
            loss.backward()
            optimizer.step()
        
            metrics['loss'] += loss.detach()
            metrics['energy_data'] += energy_data.detach()
            metrics['energy_ebm'] += energy_ebm.detach()
        torch.save(ef.state_dict(), "../weights/ef-%s" % SAVE_VERSION)
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
    import torch
    from sebm_mnist.data import load_data
#     from sebm_mnist.objectives import mle
    from sebm_mnist.modules.sgld import SGLD_sampler
    from sebm_mnist.modules.data_noise import DATA_NOISE_sampler
    from sebm_mnist.modules.energy_function import Energy_function
    
    CUDA = torch.cuda.is_available()
    if CUDA:
        DEVICE = torch.device('cuda:1')
    print('torch:', torch.__version__, 'CUDA:', CUDA)
    # optimization hyper-parameters
    num_epochs = 200
    sample_size = 1
    batch_size = 100
    lr = 2 * 1e-5
    ## model hyper-parameters
    D = 2 # data point dimensions
    hidden_dim = 1024
    pixels_dim = 28*28
    ## EBM hyper-parameters
    data_noise_std = 0.0075
    sgld_num_steps = 20
    sgld_step_size = 1
    sgld_init_sample_std = 0.1
    sgld_noise_std = 0.01
    buffer_size = 5000
    buffer_percent = 0.95
    SAVE_VERSION = 'ebm-buffer-single-sample' 
    
    ## data directory
    print('Load MNIST dataset...')
    DATA_DIR = '/home/hao/Research/sebm_data/'
    train_data, test_data = load_data(DATA_DIR, batch_size)
    
    print('Initialize energy function and optimizer...')
    ef = Energy_function(pixels_dim, hidden_dim)
    if CUDA:
        ef.cuda().to(DEVICE)   
    optimizer = torch.optim.Adam(list(ef.parameters()), lr=lr, betas=(0.9, 0.99))
    
    print('Initialize SGLD sampler...')
    sgld_sampler = SGLD_sampler(sgld_init_sample_std, sgld_noise_std, CUDA, DEVICE)
    
    print('Initialize data noise sampler...')
    data_noise_sampler = DATA_NOISE_sampler(data_noise_std, CUDA, DEVICE)
#     data_noise_sampler = None
    
    train(optimizer=optimizer, 
          ef=ef, 
          sgld_sampler=sgld_sampler,
          data_noise_sampler=data_noise_sampler,
          train_data=train_data, 
          num_epochs=num_epochs, 
          sample_size=sample_size, 
          sgld_num_steps=sgld_num_steps, 
          sgld_step_size=sgld_step_size,
          buffer_size=buffer_size, 
          buffer_percent=buffer_percent,
          CUDA=CUDA, 
          DEVICE=DEVICE, 
          SAVE_VERSION=SAVE_VERSION)
    
    