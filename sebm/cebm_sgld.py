import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import time
"""
training procedure for conjugate EBM with latent variable z, using SGLD for sampling from model distribution.
"""
class SGLD_sampler():
    """
    An sampler using stochastic gradient langevin dynamics 
    """
    def __init__(self, noise_std, lr, buffer_size, buffer_percent, grad_clipping, device):
        super(self.__class__, self).__init__()
        self.initial_dist = Uniform(-1 * torch.ones(1).cuda().to(device),
                                   torch.ones(1).cuda().to(device))
                        
        self.lr = lr
        self.noise_std = noise_std
        self.buffer_size = buffer_size
        self.buffer_percent = buffer_percent
        self.persistent_samples = None
        self.grad_clipping=grad_clipping
    
    def sgld_update(self, ebm, batch_size, pixels_size, num_steps, persistent=True):
        """
        perform update using slgd
        """
        
        if persistent: # persistent sgld
            if self.persistent_samples is None: # if not yet initialized
                self.persistent_samples = self.init_samples(self.buffer_size, pixels_size)
                assert self.persistent_samples.shape == (self.buffer_size, 1, pixels_size, pixels_size), "ERROR! buffer sampels have expected shape."
            num_from_buffer = int(self.buffer_percent * batch_size)
            num_from_init = batch_size - num_from_buffer
            samples_from_buffer = self.sample_from_buffer(num_from_buffer)
            
            samples_from_init = self.init_samples(num_from_init, pixels_size)
            samples = torch.cat((samples_from_buffer, samples_from_init), 0)
            assert samples.shape == (batch_size, 1, pixels_size, pixels_size), "ERROR! samples have unexpected shape."
    
        else: 
            samples = self.init_samples(batch_size, pixels_size)
        
        for l in range(num_steps):
            # compute gradient 
            samples.requires_grad = True
            grads = torch.autograd.grad(outputs=ebm.energy(samples).sum(), inputs=samples)[0]
            if self.grad_clipping:
                grads = torch.clamp(grads, min=-1e-2, max=1e-2)
            noise = self.noise_std * torch.randn_like(grads)
            samples = (samples - (self.lr / 2) * grads + noise).detach()
        if persistent:
            self.persistent_samples = torch.cat((samples[:num_from_buffer], self.persistent_samples[num_from_buffer:]), 0)
            assert self.persistent_samples.shape == (self.buffer_size, 1, pixels_size, pixels_size), "ERROR! buffer samples have unexpected shape."
        return samples
        
    def init_samples(self, batch_size, pixels_size):
        return self.initial_dist.sample((batch_size, 1, pixels_size, pixels_size,)).squeeze(-1)
    
    def sample_from_buffer(self, sample_size):
        rand_index = torch.randperm(self.persistent_samples.shape[0])
        self.persistent_samples = self.persistent_samples[rand_index]
        return self.persistent_samples[:sample_size]
        
class Train_procedure():
    def __init__(self, optimizer, ebm, sgld_sampler, sgld_num_steps, data_noise_std, train_data, num_epochs, batch_size, regularize_factor, device, save_version):
        super(self.__class__, self).__init__()
        self.optimizer = optimizer
        self.ebm = ebm
        self.sgld_sampler = sgld_sampler
        self.sgld_num_steps = sgld_num_steps
        self.data_noise_std = data_noise_std
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.reg_alpha = regularize_factor
        self.device = device
        self.save_version = save_version
        
    def train(self):
        for epoch in range(self.num_epochs):
            time_start = time.time()
            metrics = dict()
            for b, (images, _) in enumerate(self.train_data):
                self.optimizer.zero_grad()
                images = images.cuda().to(self.device)
                images = images + self.data_noise_std * torch.randn_like(images)
                trace = self.pcd(images)
                trace['loss'].backward()
                self.optimizer.step()
                for key in trace.keys():
                    if key not in metrics:
                        metrics[key] = trace[key].detach()
                    else:
                        metrics[key] += trace[key].detach()   
            torch.save(ebm.state_dict(), "../weights/ebm-%s" % self.save_version)
            self.logging(metrics=metrics, N=b+1, epoch=epoch)
            time_end = time.time()
            print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, self.num_epochs, time_end - time_start))

    def pcd(self, images_data):
        """
        we acquire samples from ebm using stochastic gradient langevin dynamics
        """ 
        trace = dict()
        batch_size, C, pixels_size, _ = images_data.shape
        energy_data = self.ebm.energy(images_data)
        images_ebm = self.sgld_sampler.sgld_update(ebm=ebm, 
                                                      batch_size=batch_size, 
                                                      pixels_size=pixels_size, 
                                                      num_steps=self.sgld_num_steps, 
                                                      persistent=True)
        energy_ebm = ebm.energy(images_ebm)
        trace['loss'] = (energy_data - energy_ebm).mean() + self.reg_alpha * (energy_data**2).mean()
        trace['energy_data'] = energy_data.detach().mean()
        trace['energy_ebm'] = energy_ebm.detach().mean()
        return trace
    
    
    def logging(self, metrics, N, epoch):
        if epoch == 0:
            log_file = open('../results/log-' + self.save_version + '.txt', 'w+')
        else:
            log_file = open('../results/log-' + self.save_version + '.txt', 'a+')
        metrics_print = ",  ".join(['%s=%.3e' % (k, v / N) for k, v in metrics.items()])
        print("Epoch=%d, " % (epoch+1) + metrics_print, file=log_file)
        log_file.close()

    
if __name__ == "__main__":
    import torch
    import argparse
    from sebm.data import load_mnist  
    from sebm.models import CEBM_1ss
    parser = argparse.ArgumentParser('Conjugate EBM')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default=0, type=int)
#     parser.add_argument('--exp_name', default=None)
    ## data config
    parser.add_argument('--dataset', required=True, choices=['mnist'])
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--data_noise_std', default=1e-2, type=float)
    ## optim config
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--optimize_priors', default=False, type=bool)
    ## arch config
    parser.add_argument('--arch', default='simplenet', choices=['simplenet'])
    parser.add_argument('--channels', default="[64, 64, 32, 32]")
    parser.add_argument('--kernels', default="[3, 4, 4, 4]")
    parser.add_argument('--strides', default="[1, 2, 2, 2]")
    parser.add_argument('--paddings', default="[1, 1, 1, 1]")
    parser.add_argument('--hidden_dim', default="[128]")
    parser.add_argument('--latent_dim', default=10, type=int)
    parser.add_argument('--activation', default='Swish')
    parser.add_argument('--leak', default=0.01, type=float)
    ## training config
    parser.add_argument('--num_epochs', default=200, type=int)
    ## sgld sampler config
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--buffer_percent', default=0.95, type=float)
    parser.add_argument('--sgld_noise_std', default=7.5e-3, type=float)
    parser.add_argument('--sgld_lr', default=1.0, type=float)
    parser.add_argument('--sgld_num_steps', default=50, type=int)
    ## regularization config
    parser.add_argument('--regularize_factor', default=1e-3, type=float)
    args = parser.parse_args()
    
    device = torch.device('cuda:%d' % args.device)
    save_version = 'cebm-dataset=%s-seed=%d-lr=%s-latentdim=%d-data_noise_std=%s-sgld_noise_std=%s-sgld_lr=%s-sgld_num_steps=%s-buffer_size=%d-buffer_percent=%.2f-reg_alpha=%s-act=%s-arch=%s' % (args.dataset, args.seed, args.lr, args.latent_dim, args.data_noise_std, args.sgld_noise_std, args.sgld_lr, args.sgld_num_steps, args.buffer_size, args.buffer_percent, args.regularize_factor, args.activation, args.arch)

    print('Experiment with ' + save_version)
    if args.dataset == 'mnist':
        print('Load MNIST dataset...')
        im_height, im_width, input_channels = 28, 28, 1
        train_data, test_data = load_mnist(args.data_dir, args.batch_size, normalizing=0.5, resize=None)
    else:
        raise NotImplementError
    print('Initialize EBM...')
    ebm = CEBM_1ss(arch=args.arch,
                  optimize_priors=args.optimize_priors,
                  device=device,
                  im_height=im_height, 
                  im_width=im_width, 
                  input_channels=input_channels, 
                  channels=eval(args.channels), 
                  kernels=eval(args.kernels), 
                  strides=eval(args.strides), 
                  paddings=eval(args.paddings), 
                  hidden_dim=eval(args.hidden_dim),
                  latent_dim=args.latent_dim,
                  activation=args.activation)
    ebm = ebm.cuda().to(device)
    optimizer = getattr(torch.optim, args.optimizer)(list(ebm.parameters()), lr=args.lr)
    
    print('Initialize sgld sampler...')
    sgld_sampler = SGLD_sampler(noise_std=args.sgld_noise_std,
                                lr=args.sgld_lr,
                                buffer_size=args.buffer_size,
                                buffer_percent=args.buffer_percent,
                                grad_clipping=False,
                                device=device)
    
    print('Start training...')
    trainer = Train_procedure(optimizer, ebm, sgld_sampler, args.sgld_num_steps, args.data_noise_std, train_data, args.num_epochs, args.batch_size, args.regularize_factor, device, save_version)
    trainer.train()
