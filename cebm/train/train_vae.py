import torch
import torch.nn as nn
import argparse
from cebm.data import setup_data_loader
from cebm.model.vae import Encoder, Decoder, Decoder_GMM, parse_decoder_network_args
from cebm.utils import set_seed, create_exp_name
from cebm.train.trainer import Trainer
        
class Train_VAE(Trainer):
    def __init__(self, models, train_loader, num_epochs, device, exp_name, optimizer, lr, sample_size):
        super().__init__(models, train_loader, num_epochs, device, exp_name)
        self.sample_size = sample_size
        self.optimizer = getattr(torch.optim, optimizer)(list(self.models['enc'].parameters())+list(self.models['dec'].parameters()), lr=lr)
        self.metric_names = ['ELBO', 'LL', 'KL']

    def train_epoch(self, epoch):
        enc = self.models['enc']
        dec = self.models['dec']
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        for b, (images, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            images = images.repeat(self.sample_size, 1, 1, 1, 1).to(self.device)
            loss, metric_epoch = self.loss(enc, dec, images, metric_epoch)
            loss.backward()
            self.optimizer.step()
        return {k: (v / (b+1)).item() for k, v in metric_epoch.items()}
    
    def loss(self, enc, dec, x, metric_epoch):
        z, log_qz = enc.forward(x)
        recon, ll, log_pz = dec.forward(z, x)
        kl = log_qz - log_pz
        elbo = (ll - kl).mean()
        metric_epoch['ELBO'] += elbo.detach()
        metric_epoch['LL'] += ll.mean().detach()
        metric_epoch['KL'] += kl.mean().detach()
        return - elbo.mean(), metric_epoch

def init_models(model_name, device, model_args, enc_network_args, dec_network_args):
    if model_name == 'VAE':
        enc = Encoder(**enc_network_args)   
        dec = Decoder(model_args['optimize_prior'], model_args['device'], **dec_network_args)
    elif model_name == 'VAE_GMM':
        enc = Encoder(**enc_network_args)   
        dec = Decoder_GMM(model_args['optimize_prior'], model_args['device'], model_args['num_clusters'], **dec_network_args)
    return {'enc': enc.to(device), 'dec': dec.to(device)}

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset_args = {'data': args.data, 
                    'data_dir': args.data_dir, 
                    'num_shots': -1,
                    'batch_size': args.batch_size,
                    'train': True, 
                    'normalize': False}
    train_loader, im_h, im_w, im_channels = setup_data_loader(**dataset_args)
    
    enc_network_args = {'reparameterized': True,
                        'im_height': im_h, 
                        'im_width': im_w, 
                        'input_channels': im_channels, 
                        'channels': eval(args.enc_channels), 
                        'kernels': eval(args.enc_kernels), 
                        'strides': eval(args.enc_strides), 
                        'paddings': eval(args.enc_paddings),
                        'activation': args.activation,
                        'hidden_dim': eval(args.hidden_dim),
                        'latent_dim': args.latent_dim}
    
    dec_network_args = parse_decoder_network_args(im_h, im_w, im_channels, args)
    model_args = {'optimize_prior': args.optimize_prior,
                  'device': device,
                  'num_clusters': args.num_clusters}
    
    models = init_models(args.model_name, device, model_args, enc_network_args, dec_network_args)
    exp_name = create_exp_name(args)
    print("Experiment: %s" % exp_name)
    
    trainer_args = {'models': models,
                    'train_loader': train_loader,
                    'num_epochs': args.num_epochs,
                    'device': device,
                    'exp_name': exp_name,
                    'optimizer': args.optimizer,
                    'lr': args.lr,
                    'sample_size': args.sample_size}
    trainer = Train_VAE(**trainer_args)
    print('Start Training..')
    trainer.train()
        
def parse_args():
    parser = argparse.ArgumentParser('VAE')
    parser.add_argument('--model_name', required=True, choices=['VAE', 'VAE_GMM'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--exp_id', default=None)
    ## data config
    parser.add_argument('--data', required=True, choices=['mnist', 'cifar10', 'svhn', 'fashionmnist'])
    parser.add_argument('--data_dir', default='../datasets/', type=str)
    ## optim config
    parser.add_argument('--optimizer', choices=['AdamW', 'Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--optimize_prior', default=False, action='store_true')
    ## arch config
    parser.add_argument('--enc_channels', default="[64,64,32,32]")
    parser.add_argument('--enc_kernels', default="[3,4,4,4]")
    parser.add_argument('--enc_strides', default="[1,2,2,2]")
    parser.add_argument('--enc_paddings', default="[1,1,1,1]")
    parser.add_argument('--dec_channels', default="[32,32,64,64]")
    parser.add_argument('--dec_kernels', default="[4,4,4,3]")
    parser.add_argument('--dec_strides', default="[2,2,2,1]")
    parser.add_argument('--dec_paddings', default="[1,1,0,0]")
    parser.add_argument('--hidden_dim', default="[128]")
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--activation', default='ReLU')
    parser.add_argument('--num_clusters', default=20, type=int)
    ## training config
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--sample_size', default=1, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
