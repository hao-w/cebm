import torch
import time
from ffebm.objectives import marginal_kl_multilayers

def train(optimizer, ebms, proposals, train_data, num_epochs, sample_size, batch_size, num_patches, reg_alpha, CUDA, DEVICE, SAVE_VERSION):
    """
    training an energy based model (ebm) and a proposal jointly
    by sampling z from the conjugate posterior.
    """
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = dict()
#         metrics = {'loss_theta' : 0.0, 'loss_phi' : 0.0, 'ess' : 0.0}
        for b, (images, _) in enumerate(train_data):
#             pixels_size = images.shape[-1]
#             batch_size = images.shape[0]
            optimizer.zero_grad()
            if CUDA:
                images = images.cuda().to(DEVICE)
#             if data_noise_sampler is not None: ## add Gaussian noise to true data images
#                 data_noise = data_noise_sampler.sample(batch_size, mnist_size)
#                 assert images.shape == data_noise.shape, "ERROR! data noise have unexpected shape."
#                 images = images + data_noise
            trace = marginal_kl_multilayers(ebms, proposals, images, sample_size, num_patches, reg_alpha)
            (trace['loss_phi1']+trace['loss_theta1']+trace['loss_phi2']+trace['loss_theta2']+trace['loss_phi3']+trace['loss_theta3']).backward()
            optimizer.step()
            for key in trace.keys():
                if key not in metrics:
                    metrics[key] = trace[key].detach()
                else:
                    metrics[key] += trace[key].detach()   
            print('pass!')
        torch.save(ebm1.state_dict(), "../weights/ebm1-%s" % SAVE_VERSION)
        torch.save(proposal1.state_dict(), "../weights/proposal1-%s" % SAVE_VERSION)
        torch.save(ebm2.state_dict(), "../weights/ebm2-%s" % SAVE_VERSION)
        torch.save(proposal2.state_dict(), "../weights/proposal2-%s" % SAVE_VERSION)
        torch.save(ebm3.state_dict(), "../weights/ebm3-%s" % SAVE_VERSION)
        torch.save(proposal3.state_dict(), "../weights/proposal3-%s" % SAVE_VERSION)
        
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
#     from ffebm.data_noise import DATA_NOISE_sampler
    from ffebm.nets.ffebm_multilayers import Energy_function
    from ffebm.nets.proposal_onelayer import Proposal
    
    CUDA = torch.cuda.is_available()
    if CUDA:
        DEVICE = torch.device('cuda:1')
    print('torch:', torch.__version__, 'CUDA:', CUDA)

    num_epochs = 1000
    batch_size = 100
    sample_size = 10
    latent_dim1 = 32
    latent_dim2 = 32
    latent_dim3 = 32
    hidden_dim = 128
    mnist_size = 28
    patch_size1 = 4
#     num_patches1 = 28 - patch_size1 + 1
    patch_size2 = 3
#     num_patches2 = 7 - patch_size2 + 1
    patch_size3 = 3
#     num_patches3 = 1
    lr = 5 * 1e-4
    ## EBM hyper-parameters
    data_noise_std = 1.5e-2
    reg_alpha = 0.0
    SAVE_VERSION = 'mnist-ffebm-2layers'
    
    ## data directory
    print('Load MNIST dataset...')
    DATA_DIR = '../../../sebm_data/'
    train_data, test_data = load_mnist(DATA_DIR, batch_size, normalizing=None, resize=None)
    
    print('Initialize EBM, proposal and optimizer...')
    ebm1 = Energy_function(in_channel=1, out_channel=latent_dim1, kernel_size=4, stride=4, padding=0, CUDA=CUDA, DEVICE=DEVICE)
    ebm2 = Energy_function(in_channel=latent_dim1, out_channel=latent_dim2, kernel_size=3, stride=3, padding=1, CUDA=CUDA, DEVICE=DEVICE)
    ebm3 = Energy_function(in_channel=latent_dim3, out_channel=latent_dim3, kernel_size=3, stride=1, padding=0, CUDA=CUDA, DEVICE=DEVICE)
    proposal1 = Proposal(latent_dim1, hidden_dim, patch_size1**2, in_channel=1)
    proposal2 = Proposal(latent_dim2, hidden_dim, patch_size2**2, in_channel=32)
    proposal3 = Proposal(latent_dim3, hidden_dim, patch_size3**2, in_channel=32)
    if CUDA:
        with torch.cuda.device(DEVICE):
            ebm1.cuda()
            ebm2.cuda()
            ebm3.cuda()
            proposal1.cuda()
            proposal2.cuda()
            proposal3.cuda()
    optimizer = torch.optim.Adam(list(ebm1.parameters())+list(proposal1.parameters())+list(ebm2.parameters())+list(proposal2.parameters())+list(ebm3.parameters())+list(proposal3.parameters()), lr=lr, betas=(0.9, 0.999))

#     print('Initialize data noise sampler...')
#     if data_noise_std == 0.0:
#         data_noise_sampler = None
#     elif data_noise_std > 0:
#         data_noise_sampler = DATA_NOISE_sampler(data_noise_std, CUDA, DEVICE)
#     else:
#         raise ValueError
        
    print('Start training...')
    train(optimizer=optimizer, 
          ebms=(ebm1, ebm2, ebm3), 
          proposals=(proposal1, proposal2, proposal3),
          train_data=train_data, 
          num_epochs=num_epochs,
          sample_size=sample_size,
          batch_size=batch_size,
          num_patches=(7, 3, 1),
          reg_alpha=reg_alpha,
          CUDA=CUDA, 
          DEVICE=DEVICE, 
          SAVE_VERSION=SAVE_VERSION)
    
    
