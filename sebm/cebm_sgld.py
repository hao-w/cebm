import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import time
from sebm.models import CEBM_1ss, CEBM_2ss
"""
training procedure for conjugate EBM with latent variable z, using SGLD for sampling from model distribution.
"""
class SGLD_sampler():
    """
    An sampler using stochastic gradient langevin dynamics 
    """
    def __init__(self, device, input_channels, noise_std, lr, pixel_size, buffer_size, buffer_percent, buffer_init, buffer_dup_allowed, grad_clipping=False):
        super(self.__class__, self).__init__()
        self.initial_dist = Uniform(-1 * torch.ones((input_channels, pixel_size, pixel_size)).cuda().to(device),
                                   torch.ones((input_channels, pixel_size, pixel_size)).cuda().to(device))
          
        self.lr = lr
        self.noise_std = noise_std
        self.buffer_size = buffer_size
        self.buffer_percent = buffer_percent
        self.grad_clipping=grad_clipping
        self.buffer_init = buffer_init
        if self.buffer_init: # whether initialize buffer at the beginning 
            self.buffer = self.initial_dist.sample((self.buffer_size, ))
            self.buffer_dup_allowed = buffer_dup_allowed
        else:
            self.buffer = None
            self.buffer_dup_allowed = True # without init buffer, always allow duplicated sampling, i.e. ignore that parameter    
        self.device = device

    def sample_from_buffer(self, batch_size):
        """
        sample from buffer with a frequency
        self.buffer_dup_allowed = True allows sampling the same chain multiple time within one sampling step
        which is used in JEM and IGEBM  
        """
        if self.buffer_dup_allowed:
            samples = self.initial_dist.sample((batch_size, ))
            inds = torch.randint(0, self.buffer_size, (batch_size, ), device=self.device)
            samples_from_buffer = self.buffer[inds]
            rand_mask = (torch.rand(batch_size, device=self.device) < self.buffer_percent)
            samples[rand_mask] = samples_from_buffer[rand_mask]
        else:
            inds = int(self.buffer_percent * batch_size)
            self.buffer = self.buffer[torch.randperm(len(self.buffer))]
            samples_from_buffer = self.buffer[:inds]
            samples_from_init = self.initial_dist.sample((batch_size - inds,))
            samples = torch.cat((samples_from_buffer, samples_from_init), 0)
        assert samples.shape[0] == batch_size, "Samples have unexpected shape."            
        return samples, inds

    def nsgd_steps(self, ebm, samples, num_steps, logging_interval=None):
        """
        perform noisy gradient descent steps and return updated samples 
        """
        list_samples = []
        for l in range(num_steps):
            samples.requires_grad = True
            grads = torch.autograd.grad(outputs=ebm.energy(samples).sum(), inputs=samples)[0]
            if self.grad_clipping:
                grads = torch.clamp(grads, min=-1e-2, max=1e-2)
            samples = (samples - (self.lr / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
            if logging_interval is not None:
                if (l+1) % logging_interval == 0:
                    list_samples.append(samples.unsqueeze(0).detach())  
        if logging_interval is not None:
            samples = torch.cat(list_samples, 0)
        else:
            samples = samples.detach() ## added this extra detachment step, becase the last update keeps the variable in the graph somehow, need to figure out why.
        assert samples.requires_grad == False, "samples should not require gradient."
        return samples
    
    def refine_buffer(self, samples, inds):
        """
        update replay buffer
        """
        if self.buffer_init:
            if self.buffer_dup_allowed:
                self.buffer[inds] = samples
            else:
                self.buffer[:inds] = samples[:inds]
        else:
            if self.buffer is None:
                self.buffer = samples
            else:
                self.buffer = torch.cat((self.buffer, samples), 0)
                if self.buffer_size < len(self.buffer):
                    ## truncate buffer from 'head' of the queue
                    self.buffer = self.buffer[len(self.buffer) - self.buffer_size:]
                
    def sample(self, ebm, batch_size, num_steps, pcd=True, init_samples=None, logging_interval=None):
        """
        perform update using slgd
        pcd means that we sample from replay buffer (with a frequency)
        if buffer is not initialized in advance, we check its storage and 
        init from random noise if storage is smaller than batch_size.
        """
        if pcd:
            if self.buffer_init:
                samples, inds = self.sample_from_buffer(batch_size)
            else:
                if self.buffer is not None and len(self.buffer) >= batch_size:     
                    samples, _ = self.sample_from_buffer(batch_size)
                else: 
                    samples = self.initial_dist.sample((batch_size, ))
                inds = None
        else:
            if init_samples is None:
                samples = self.initial_dist.sample((batch_size, ))
            else:
                samples = init_samples
        samples = self.nsgd_steps(ebm, samples, num_steps, logging_interval=logging_interval)
        ## refine buffer if pcd
        if logging_interval is None and pcd:
            self.refine_buffer(samples, inds)
        return samples
        
class Train_procedure():
    def __init__(self, optimizer, ebm, sgld_sampler, sgld_num_steps, data_noise_std, train_data, num_epochs, regularize_factor, warmup_iters, lr, device, save_version):
        super(self.__class__, self).__init__()
        self.optimizer = optimizer
        self.ebm = ebm
        self.sgld_sampler = sgld_sampler
        self.sgld_num_steps = sgld_num_steps
        self.data_noise_std = data_noise_std
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.reg_alpha = regularize_factor
        self.warmup_iters = warmup_iters
        self.lr = lr
        self.device = device
        self.save_version = save_version
        
    def train(self):
#         cur_iter = 0.0
        for epoch in range(self.num_epochs):
            time_start = time.time()
            metrics = dict()
            for b, (images, _) in enumerate(self.train_data):
                self.optimizer.zero_grad()
                images = images.cuda().to(self.device)
                images = images + self.data_noise_std * torch.randn_like(images)
                trace = self.pcd(images)
                if trace['loss'].abs().item() > 1e+8:
                    print('Model is diverging, will terminate training..')
                    exit()
                trace['loss'].backward()
                self.optimizer.step()
                for key in trace.keys():
                    if key not in metrics:
                        metrics[key] = trace[key].detach()
                    else:
                        metrics[key] += trace[key].detach() 
            self.save_checkpoints()
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
        images_ebm = self.sgld_sampler.sample(ebm, batch_size, self.sgld_num_steps, pcd=True)
        energy_ebm = ebm.energy(images_ebm)
        trace['loss'] = (energy_data - energy_ebm).mean() + self.reg_alpha * (energy_data**2).mean()
        trace['energy_data'] = energy_data.detach().mean()
        trace['energy_ebm'] = energy_ebm.detach().mean()
        return trace
    
    
    def logging(self, metrics, N, epoch):
        if epoch == 0:
            log_file = open('results/log-' + self.save_version + '.txt', 'w+')
        else:
            log_file = open('results/log-' + self.save_version + '.txt', 'a+')
        metrics_print = ",  ".join(['%s=%.3e' % (k, v / N) for k, v in metrics.items()])
        print("Epoch=%d, " % (epoch+1) + metrics_print, file=log_file)
        log_file.close()
        
    def save_checkpoints(self):
        checkpoint_dict  = {
            'model_state_dict': self.ebm.state_dict()
#             'replay_buffer': self.sgld_sampler.buffer
            }
        torch.save(checkpoint_dict, "weights/cp-%s" % self.save_version)


def init_cebm(arch, ss, optimize_priors, device, **kwargs):
    model = eval('CEBM_%sss' % ss)
    print('Initialize Model=%s...' % model.__name__)
    ebm = model(arch, optimize_priors, device, **kwargs)
    return ebm

if __name__ == "__main__":
    import torch
    import argparse
    from sebm.data import load_data, load_mnist_heldout
    from sebm.models import CEBM_1ss, CEBM_2ss
    from util import set_seed
    parser = argparse.ArgumentParser('Conjugate EBM')
    parser.add_argument('--ss', default='2', choices=['1', '2'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default=0, type=int)
#     parser.add_argument('--exp_name', default=None)
    ## data config
    parser.add_argument('--dataset', required=True, choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet', 'celeba', 'flowers102', 'fashionmnist'])
    parser.add_argument('--data_dir', default=None, type=str)
#     parser.add_argument('--sample_size', default=1, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--data_noise_std', default=1e-2, type=float)
    ## optim config
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--optimize_priors', default=False, type=bool)
    parser.add_argument('--warmup_iters', default=1000, type=int)
    ## arch config
    parser.add_argument('--arch', default='simplenet', choices=['simplenet', 'simplenet2', 'wresnet'])
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--channels', default="[64, 64, 32, 32]")
    parser.add_argument('--kernels', default="[3, 4, 4, 4]")
    parser.add_argument('--strides', default="[1, 2, 2, 2]")
    parser.add_argument('--paddings', default="[1, 1, 1, 1]")
    parser.add_argument('--hidden_dim', default="[128]")
    parser.add_argument('--latent_dim', default=10, type=int)
    parser.add_argument('--activation', default='Swish')
    parser.add_argument('--leak', default=0.2, type=float)
    ## training config
    parser.add_argument('--num_epochs', default=200, type=int)
    ## sgld sampler config
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--buffer_percent', default=0.95, type=float)
    parser.add_argument('--buffer_init', default=False, action='store_true')
    parser.add_argument('--buffer_dup_allowed', default=False, action='store_true')
    parser.add_argument('--sgld_noise_std', default=7.5e-3, type=float)
    parser.add_argument('--sgld_lr', default=1.0, type=float)
    parser.add_argument('--sgld_num_steps', default=50, type=int)
    parser.add_argument('--grad_clipping', default=False, action='store_true')
    ## regularization config
    parser.add_argument('--regularize_factor', default=1e-3, type=float)
    parser.add_argument('--heldout_class', default=-1, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1])

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda:%d' % args.device)
    save_version = 'cebm_%sss-out=%s-d=%s-seed=%d-lr=%s-zd=%d-d_ns=%s-sgld-ns=%s-lr=%s-steps=%s-reg=%s-act=%s-arch=%s' % (args.ss, args.heldout_class, args.dataset, args.seed, args.lr, args.latent_dim, args.data_noise_std, args.sgld_noise_std, args.sgld_lr, args.sgld_num_steps, args.regularize_factor, args.activation, args.arch)
    print('Experiment with ' + save_version)
    print('Loading dataset=%s...' % args.dataset)
    if args.heldout_class == -1:
        train_data, img_dims = load_data(args.dataset, args.data_dir, args.batch_size, train=True)
    else:
        print('hold out class=%s' % args.heldout_class)
        train_data, img_dims = load_mnist_heldout(args.data_dir, args.batch_size, args.heldout_class, train=True, normalize=True)
    (input_channels, im_height, im_width) = img_dims  
    if args.arch == 'wresnet':
        ebm = init_cebm(arch=args.arch,
                        ss=args.ss,
                        optimize_priors=args.optimize_priors,
                        device=device,
                        depth=args.depth,
                        width=args.width,
                        hidden_dim=eval(args.hidden_dim),
                        latent_dim=args.latent_dim,
                        activation=args.activation,
                        leak=args.leak)
    elif args.arch == 'simplenet' or args.arch == 'simplenet2':
        ebm = init_cebm(arch=args.arch,
                        ss=args.ss,
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
                        activation=args.activation,
                        leak=args.leak)
    else:
        raise NotImplementError
        
    ebm = ebm.cuda().to(device)
    optimizer = getattr(torch.optim, args.optimizer)(list(ebm.parameters()), lr=args.lr)
    print('Initialize sgld sampler...')
    sgld_sampler = SGLD_sampler(device=device,
                                input_channels=input_channels,
                                noise_std=args.sgld_noise_std,
                                lr=args.sgld_lr,
                                pixel_size=im_height,
                                buffer_size=args.buffer_size,
                                buffer_percent=args.buffer_percent,
                                buffer_init=args.buffer_init,
                                buffer_dup_allowed=args.buffer_dup_allowed,
                                grad_clipping=args.grad_clipping)
    
    print('Start training...')
    trainer = Train_procedure(optimizer, ebm, sgld_sampler, args.sgld_num_steps, args.data_noise_std, train_data, args.num_epochs, args.regularize_factor, args.warmup_iters, args.lr, device, save_version)
    trainer.train()
