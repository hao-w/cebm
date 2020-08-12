import torch
import time
from ffebm.objectives import marginal_kl_1layer

def train(optimizer, ebm, proposal, train_data, num_epochs, sample_size, batch_size, reg_alpha, CUDA, DEVICE, SAVE_VERSION):
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
            trace = marginal_kl_1layer(ebm, proposal, images, sample_size, reg_alpha)
            loss = trace['loss_phi']+trace['loss_theta']
            if reg_alpha != 0.0:
                loss = loss + trace['regularize_term']
            loss.backward()
            optimizer.step()
            for key in trace.keys():
                if key not in metrics:
                    metrics[key] = trace[key].detach()
                else:
                    metrics[key] += trace[key].detach()   
        torch.save(ebm1.state_dict(), "../weights/ebm-%s" % SAVE_VERSION)
        torch.save(proposal1.state_dict(), "../weights/proposal-%s" % SAVE_VERSION)
 
        logging(metrics=metrics, filename=SAVE_VERSION, average_normalizer=b+1, epoch=epoch)
        time_end = time.time()
        print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, num_epochs, time_end - time_start))
        
def logging(metrics, filename, average_normalizer, epoch):
    if epoch == 0:
        log_file = open('../results/log-' + filename + '.txt', 'w+')
    else:
        log_file = open('../results/log-' + filename + '.txt', 'a+')
    metrics_print = ",  ".join(['%s=%.3e' % (k, v / average_normalizer) for k, v in metrics.items()])
    print("Epoch=%d, " % (epoch+1) + metrics_print, file=log_file)
    log_file.close()
    

    
if __name__ == "__main__":
    import torch
    from ffebm.data import load_mnist
    from ffebm.data_noise import DATA_NOISE_sampler
    from ffebm.nets.ffebm_onelayer import Energy_function
    from ffebm.nets.proposal_onelayer import Proposal
    CUDA = torch.cuda.is_available()
    if CUDA:
        DEVICE = torch.device('cuda:0')
    print('torch:', torch.__version__, 'CUDA:', CUDA)

    num_epochs = 1000
    batch_size = 100
    sample_size = 10
    latent_dim = 10
    hidden_dim = 128
    mnist_size = 28
    patch_size1 = 4
    lr = 1 * 1e-4
    ## EBM hyper-parameters
    data_noise_std = 1.5e-2
    reg_alpha = 0.01
    SAVE_VERSION = 'mnist-ffebm-1layer-reg_alpha=%.2E' % reg_alpha
    
    ## data directory
    print('Load MNIST dataset...')
    DATA_DIR = '../../../sebm_data/'
    train_data, test_data = load_mnist(DATA_DIR, batch_size, normalizing=0.5, resize=None)
    
    print('Initialize EBM, proposal and optimizer...')
    ebm1 = Energy_function(latent_dim=latent_dim, CUDA=CUDA, DEVICE=DEVICE)

    proposal1 = Proposal(latent_dim, hidden_dim, mnist_size**2)
    if CUDA:
        with torch.cuda.device(DEVICE):
            ebm1.cuda()
            proposal1.cuda()
    optimizer = torch.optim.Adam(list(ebm1.parameters())+list(proposal1.parameters()), lr=lr, betas=(0.9, 0.999))

    print('Initialize data noise sampler...')
    if data_noise_std == 0.0:
        data_noise_sampler = None
    elif data_noise_std > 0:
        data_noise_sampler = DATA_NOISE_sampler(data_noise_std, CUDA, DEVICE)
    else:
        raise ValueError
        
    print('Start training...')
    train(optimizer=optimizer, 
          ebm=ebm1, 
          proposal=proposal1,
          train_data=train_data, 
          num_epochs=num_epochs,
          sample_size=sample_size,
          batch_size=batch_size,
          reg_alpha=reg_alpha,
          CUDA=CUDA, 
          DEVICE=DEVICE, 
          SAVE_VERSION=SAVE_VERSION)
    
    
