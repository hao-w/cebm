import torch
import time
from torchvision import datasets, transforms

        
class Train_procedure():
    def __init__(self, optimizer, enc, dec, arch, train_data, num_epochs, sample_size, device, save_version):
        super(self.__class__, self).__init__()
        self.optimizer = optimizer
        self.enc = enc
        self.dec = dec
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.sample_size = sample_size
        self.device = device
        self.save_version = save_version
        self.arch = arch

    def train(self):
        for epoch in range(self.num_epochs):
            time_start = time.time()
            metrics = dict()
            for b, (images, _) in enumerate(self.train_data):
                self.optimizer.zero_grad()
                batch_size, input_c, im_h, im_w = images.shape
                if self.arch == 'mlp':
                    images = images.squeeze().view(batch_size, im_h*im_w).repeat(self.sample_size, 1, 1).cuda().to(self.device)
                elif self.arch == 'simplenet2':
                    images = images.repeat(self.sample_size, 1, 1, 1, 1).view(self.sample_size*batch_size, input_c, im_h, im_w).cuda().to(self.device)
                trace = self.elbo(images)
                (- trace['elbo']).backward()
                optimizer.step()
                for key in trace.keys():
                    if key not in metrics:
                        metrics[key] = trace[key].detach()
                    else:
                        metrics[key] += trace[key].detach()
            self.save_checkpoints()
            self.logging(metrics=metrics, N=b+1, epoch=epoch)
            time_end = time.time()
            print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, self.num_epochs, time_end - time_start))

    
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
            'enc_state_dict': self.enc.state_dict(),
            'dec_state_dict': self.dec.state_dict()
            }
        torch.save(checkpoint_dict, "weights/cp-%s" % self.save_version)

    def elbo(self, images):
        """
        compute the ELBO in vae
        """
        trace = dict()
        latents, q_log_prob = self.enc.forward(images)
        recon, ll, p_log_prob = self.dec.forward(latents, images)
        log_w = (ll + p_log_prob - q_log_prob)
        trace['elbo'] = log_w.mean()
        return trace 
    
if __name__ == "__main__":
    import torch
    import argparse
    from util import set_seed
    from sebm.models import Encoder, Decoder
    from sebm.data import load_data, load_mnist_heldout
    parser = argparse.ArgumentParser('VAE')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default=0, type=int)
    ## data config
    parser.add_argument('--dataset', required=True, choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet', 'celeba', 'flowers102'])
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--sample_size', default=1, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--reparameterized', default=False, action='store_true')
    ## optim config
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    
    parser.add_argument('--arch', default='simplenet2', choices=['simplenet2', 'mlp'])
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--channels', default="[64, 64, 32, 32]")
    parser.add_argument('--kernels', default="[3, 4, 4, 4]")
    parser.add_argument('--strides', default="[1, 2, 2, 2]")
    parser.add_argument('--paddings', default="[1, 1, 1, 1]")
    parser.add_argument('--hidden_dim1', default="[400, 200]")
    parser.add_argument('--hidden_dim2', default="[128]")
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--activation', default='ReLU')
    parser.add_argument('--heldout_class', default=-1, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1])
    ## training config
    parser.add_argument('--num_epochs', default=200, type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda:%d' % args.device)
        
    save_version = 'vae-out=%s-d=%s-seed=%s-lr=%s-zd=%s-act=%s-arch=%s' % (args.heldout_class, args.dataset, args.seed, args.lr, args.latent_dim, args.activation, args.arch) 
    ## data directory
    print('Experiment with ' + save_version)
    print('Loading dataset=%s...' % args.dataset)
    if args.heldout_class == -1:
        train_data, img_dims = load_data(args.dataset, args.data_dir, args.batch_size, train=True, normalize=False)
    else:
        print('hold out class=%s' % args.heldout_class)
        train_data, img_dims = load_mnist_heldout(args.data_dir, args.batch_size, args.heldout_class, train=True, normalize=False)
    (input_channels, im_height, im_width) = img_dims  
    
    print('Initialize VAE...')
    if args.arch == 'simplenet2':
        enc = Encoder(arch=args.arch,
                      reparameterized=args.reparameterized,
                      im_height=im_height, 
                      im_width=im_width, 
                      input_channels=input_channels, 
                      channels=eval(args.channels), 
                      kernels=eval(args.kernels), 
                      strides=eval(args.strides), 
                      paddings=eval(args.paddings), 
                      hidden_dim=eval(args.hidden_dim2),
                      latent_dim=args.latent_dim,
                      activation=args.activation,
                      leak=None)
        dec = Decoder(arch=args.arch,
                      device=args.device,
                      input_channels=input_channels, 
                      channels=eval(args.channels), 
                      kernels=eval(args.kernels), 
                      strides=eval(args.strides), 
                      paddings=eval(args.paddings), 
                      output_paddings=[1,0,0,0], ## TODO: hand-coded for now
                      hidden_dim=eval(args.hidden_dim2),
                      latent_dim=args.latent_dim,
                      output_dim=288, ## TODO: hand-coded for now
                      activation=args.activation,
                      leak=None)
    elif args.arch =='mlp':
        enc = Encoder(arch=args.arch, 
                      reparameterized=args.reparameterized,
                      input_dim=im_height*im_width,
                      hidden_dim1=eval(args.hidden_dim1), 
                      hidden_dim2=eval(args.hidden_dim2),
                      output_dim=args.latent_dim, 
                      activation=args.activation)
        dec = Decoder(arch=args.arch, 
                      device=args.device,
                      input_dim=args.latent_dim,
                      hidden_dim1=eval(args.hidden_dim1), 
                      hidden_dim2=eval(args.hidden_dim2),
                      output_dim=im_height*im_width, 
                      activation=args.activation)   
    else:
        raise NotImplementError
    
    
    enc = enc.cuda().to(device)
    dec = dec.cuda().to(device)
    optimizer = getattr(torch.optim, args.optimizer)(list(enc.parameters())+list(dec.parameters()), lr=args.lr)

    print('Start training...')
    trainer = Train_procedure(optimizer, enc, dec, args.arch, train_data, args.num_epochs, args.sample_size, device, save_version)
    trainer.train()
    
    
