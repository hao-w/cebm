import torch
import time
from ffebm.objectives import mle
from ffebm.data import load_mnist
from ffebm.data_noise import DATA_NOISE_sampler
from ffebm.sgld import SGLD_sampler
from ffebm.nets.conjugate_vanilla_ebm import Energy_function

def train(optimizer, ebm, data_noise_sampler, sgld_sampler, sgld_num_steps, train_data, num_epochs, batch_size, reg_alpha, CUDA, DEVICE, SAVE_VERSION):
    """
    training an energy based model (ebm) and a proposal jointly
    by sampling z from the conjugate posterior.
    """
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = dict()
        for b, (images, _) in enumerate(train_data):

            optimizer.zero_grad()
            if CUDA:
                images = images.cuda().to(DEVICE)
            if data_noise_sampler is not None: ## add Gaussian noise to true data images
                data_noise = data_noise_sampler.sample(batch_size, 28)
                assert images.shape == data_noise.shape, "ERROR! data noise have unexpected shape."
                images = images + data_noise
            trace = mle(ebm, sgld_sampler, sgld_num_steps, images, reg_alpha)
            loss = trace['loss_theta']
            if reg_alpha != 0.0:
                loss = loss + trace['regularize_term']   
            loss.backward()
            optimizer.step()
            for key in trace.keys():
                if key not in metrics:
                    metrics[key] = trace[key].detach()
                else:
                    metrics[key] += trace[key].detach()   
        torch.save(ebm.state_dict(), "../weights/grid_search/ebm-%s" % SAVE_VERSION)
 
        logging(metrics=metrics, filename=SAVE_VERSION, average_normalizer=b+1, epoch=epoch)
        time_end = time.time()
        print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, num_epochs, time_end - time_start))
        
def logging(metrics, filename, average_normalizer, epoch):
    if epoch == 0:
        log_file = open('../results/grid_search/log-' + filename + '.txt', 'w+')
    else:
        log_file = open('../results/grid_search/log-' + filename + '.txt', 'a+')
    metrics_print = ",  ".join(['%s=%.3e' % (k, v / average_normalizer) for k, v in metrics.items()])
    print("Epoch=%d, " % (epoch+1) + metrics_print, file=log_file)
    log_file.close()
    
def grid_search_training(num_epochs, batch_size, latent_dim, lr, data_noise_std, sgld_noise_std, sgld_step_size, sgld_num_steps, buffer_size, buffer_percent, reg_alpha, CUDA, DEVICE):
  
    mnist_size = 28
    ## EBM hyper-parameters
    data_noise_std = 3e-2
    sgld_noise_std = 7.5e-3
    sgld_step_size = 1
    sgld_num_steps = 5
    buffer_size = 5000
    buffer_percent = 0.95
    reg_alpha = 0.01
    SAVE_VERSION = 'mnist-conjugate_sgld-lr=%.2E-data_noise_std=%.2E-sgld_noise_std=%.2E-sgld_step_size=%.2E-sgld_num_steps=%.2E-buffer_size=%d-buffer_percent=%.2f-reg_alpha=%.2E' % (lr, data_noise_std, sgld_noise_std, sgld_step_size, sgld_num_steps, buffer_size, buffer_percent,reg_alpha)
    ## data directory
    print('Load MNIST dataset...')
    DATA_DIR = '../../../sebm_data/'
    train_data, test_data = load_mnist(DATA_DIR, batch_size, normalizing=0.5, resize=None)
    
    print('Initialize data noise sampler...')
    if data_noise_std == 0.0:
        data_noise_sampler = None
    elif data_noise_std > 0:
        data_noise_sampler = DATA_NOISE_sampler(data_noise_std, CUDA, DEVICE)
    else:
        raise ValueError
        
    print('Initialize slgd sampler...')
    sgld_sampler = SGLD_sampler(noise_std=sgld_noise_std,
                                step_size=sgld_step_size,
                                buffer_size=buffer_size,
                                buffer_percent=buffer_percent,
                                grad_clipping=False,
                                CUDA=CUDA,
                                DEVICE=DEVICE)

    print('Initialize EBM and optimizer...')
    ebm = Energy_function(latent_dim=latent_dim, CUDA=CUDA, DEVICE=DEVICE)
    if CUDA:
        with torch.cuda.device(DEVICE):
            ebm.cuda()
    optimizer = torch.optim.Adam(list(ebm.parameters()), lr=lr, betas=(0.9, 0.999))

    
    print('Start training...')
    train(optimizer=optimizer, 
          ebm=ebm, 
          data_noise_sampler=data_noise_sampler,
          sgld_sampler=sgld_sampler,
          sgld_num_steps=sgld_num_steps,
          train_data=train_data, 
          num_epochs=num_epochs,
          batch_size=batch_size,
          reg_alpha=reg_alpha,
          CUDA=CUDA, 
          DEVICE=DEVICE, 
          SAVE_VERSION=SAVE_VERSION)
    
if __name__ == "__main__":
    import torch
    CUDA = torch.cuda.is_available()
    if CUDA:
        DEVICE = torch.device('cuda:1')
    print('torch:', torch.__version__, 'CUDA:', CUDA)
    DNSs = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2]
    SGLDNSs = [1e-2, 7.5e-3, 5e-3, 2.5e-3]
    REGs = [0.0, 1e-2, 5e-2, 1e-1]
    for i in DNSs:
        for j in SGLDNSs:
            for k in REGs:
                print('grid search with data_noise_std=%.2E, sgld_noise_std=%.2E, regular=%.2E...' % (i, j, k))
                grid_search_training(num_epochs=50, 
                                     batch_size=100, 
                                     latent_dim=10, 
                                     lr=1e-4, 
                                     data_noise_std=i, 
                                     sgld_noise_std=j, 
                                     sgld_step_size=1, 
                                     sgld_num_steps=50, 
                                     buffer_size=5000, 
                                     buffer_percent=0.95, 
                                     reg_alpha=k, 
                                     CUDA=CUDA, 
                                     DEVICE=DEVICE)
    
    
    
    
