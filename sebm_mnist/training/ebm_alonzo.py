import torch
import time
from sebm_mnist.objectives import mle

def train(optimizer, ef, sgld_sampler, data_noise_sampler, train_data, num_epochs, sgld_num_steps, sgld_step_size, buffer_size, buffer_percent, regularize_alpha, CUDA, DEVICE, SAVE_VERSION):
    """
    training the energy based model (ebm) by maximizing the marginal likelihood
    """
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = {'loss' : 0.0}
        for b, (images, _) in enumerate(train_data):
            pixels_size = images.shape[-1]
            batch_size = images.shape[0]
            optimizer.zero_grad()
#             images = images.squeeze(1).view(-1, pixels_size)
#             images = images.repeat(sample_size, 1, 1)
            if CUDA:
                images = images.cuda().to(DEVICE)
            if data_noise_sampler is not None: ## add Gaussian noise to true data images
                data_noise = data_noise_sampler.sample(batch_size, pixels_size)
                assert images.shape == data_noise.shape, "ERROR! data noise have unexpected shape."
                images = images + 2 * data_noise
            energy_data, energy_ebm = mle(ef, sgld_sampler, images, sgld_num_steps, sgld_step_size, buffer_size, buffer_percent)
            loss = energy_data.mean() - energy_ebm.mean()
            if regularize_alpha is not None:
                loss = loss + regularize_alpha * ((energy_data**2).mean() + (energy_ebm**2).mean())
            loss.backward()
            optimizer.step()
        
            metrics['loss'] += loss.detach()
#             metrics['energy_data'] += energy_data.mean().detach()
#             metrics['energy_ebm'] += energy_ebm.mean().detach()
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
    from sebm_mnist.modules.sgld import SGLD_sampler
    from sebm_mnist.modules.data_noise import DATA_NOISE_sampler
    from sebm_mnist.modules.energy_function_anatomy import Energy_function
    
    torch.manual_seed(123)
    
    CUDA = torch.cuda.is_available()
    if CUDA:
        DEVICE = torch.device('cuda:0')
    print('torch:', torch.__version__, 'CUDA:', CUDA)
    # optimization hyper-parameters
    num_epochs = 200
    batch_size = 100
    lr = 5 * 1e-5
    ## EBM hyper-parameters
    sgld_num_steps = 40
    sgld_noise_std = 7.5e-3
    sgld_step_size = 1
    data_noise_std = 1.5e-2
    buffer_size = 5000
    buffer_percent = 0.95
    regularize_alpha = 0.01
    clipping_sgld = False
    SAVE_VERSION = 'ebm-anatomy-lr=%.2E-data-std=%.2E-sgld-steps=%.2E-size=%.2E-std=%.2E' % (lr, data_noise_std, sgld_num_steps, sgld_step_size, sgld_noise_std) 
    
    ## data directory
    print('Load MNIST dataset...')
    DATA_DIR = '/home/hao/Research/sebm_data/'
    train_data, test_data = load_data(DATA_DIR, batch_size)
    
    print('Initialize energy function and optimizer...')
    ef = Energy_function()
    if CUDA:
        ef.cuda().to(DEVICE)   
    optimizer = torch.optim.Adam(list(ef.parameters()), lr=lr, betas=(0.9, 0.999))
    
    print('Initialize SGLD sampler...')
    sgld_sampler = SGLD_sampler(sgld_noise_std, clipping_sgld, CUDA, DEVICE)
    
    print('Initialize data noise sampler...')
    data_noise_sampler = DATA_NOISE_sampler(data_noise_std, CUDA, DEVICE)
#     data_noise_sampler = None
    
    train(optimizer=optimizer, 
          ef=ef, 
          sgld_sampler=sgld_sampler,
          data_noise_sampler=data_noise_sampler,
          train_data=train_data, 
          num_epochs=num_epochs, 
          sgld_num_steps=sgld_num_steps, 
          sgld_step_size=sgld_step_size,
          buffer_size=buffer_size, 
          buffer_percent=buffer_percent,
          regularize_alpha=regularize_alpha,
          CUDA=CUDA, 
          DEVICE=DEVICE, 
          SAVE_VERSION=SAVE_VERSION)
    
    
