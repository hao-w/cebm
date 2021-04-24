import torch
import torch.nn as nn
import argparse
from cebm.data import setup_data_loader
from cebm.model.gan import Generator_BIGAN, Generator_BIGAN_GMM, Discriminator_BIGAN, Encoder_BIGAN
from cebm.utils import set_seed, create_exp_name
from cebm.train.trainer import Trainer

class Train_BIGAN(Trainer):
    def __init__(self, models, train_loader, num_epochs, device, exp_name, optimizer, lr, image_noise_std):
        super().__init__(models, train_loader, num_epochs, device, exp_name)
        self.image_noise_std = image_noise_std
        self.optimizerD = getattr(torch.optim, optimizer)(list(self.models['disc'].parameters()), lr=lr, betas=(0.5, 0.999))
        self.optimizerGE = getattr(torch.optim, optimizer)(list(self.models['gen'].parameters())+list(self.models['enc'].parameters()), lr=args.lr,betas=(0.5, 0.999))
        self.bceloss = nn.BCELoss()
        self.metric_names = ['lossD', 'lossG', 'D(x)', 'D(G(z))']
        
    def reset_grad(self):
        for k, m in self.models.items():
            m.zero_grad()

    def train_epoch(self, epoch):
        disc = self.models['disc']
        gen = self.models['gen']
        enc = self.models['enc']
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        noise_std = 0.1 * (self.num_epochs - epoch) / self.num_epochs
        for b, (images, _) in enumerate(self.train_loader): 
            real_images = images.to(self.device)
            fake_z, fake_images = gen.forward(real_images.shape[0])
            real_z = enc.forward(real_images)
            noise_real = torch.randn_like(real_images) * noise_std
            noise_fake = torch.randn_like(fake_images) * noise_std
            # discrinimate real and fake examples
            pred_fake = disc.binary_pred(fake_images + noise_fake, fake_z)
            pred_real = disc.binary_pred(real_images + noise_real, real_z)
            # update GE
            self.optimizerGE.zero_grad()
            real_label = torch.ones(len(images), device=self.device)
            fake_label = torch.zeros(len(images), device=self.device)
            loss_ge = self.bceloss(pred_fake, real_label) + self.bceloss(pred_real, fake_label)
            loss_ge.backward()
            self.optimizerGE.step()
            # update D
            self.optimizerD.zero_grad()    
            # discrinimate real and fake examples
            pred_fake = disc.binary_pred(fake_images.detach() + noise_fake, fake_z.detach())
            pred_real = disc.binary_pred(real_images + noise_real, real_z.detach())
            loss_d = self.bceloss(pred_real, real_label) + self.bceloss(pred_fake, fake_label)
            loss_d.backward()
            self.optimizerD.step()
            self.reset_grad()
            metric_epoch['lossD'] += loss_d.detach()
            metric_epoch['lossG'] += loss_ge.detach()
            metric_epoch['D(x)'] += pred_real.detach().mean()
            metric_epoch['D(G(z))'] += pred_fake.detach().mean()
        return {k: (v / (b+1)).item() for k, v in metric_epoch.items()}

def init_models(model_name, device, model_args, disc_enc_network_args, gen_network_args):
    if model_name == 'BIGAN':
        disc = Discriminator_BIGAN(**disc_enc_network_args)   
        enc = Encoder_BIGAN(**disc_enc_network_args)
        gen = Generator_BIGAN(**gen_network_args)
    elif model_name == 'BIGAN_GMM':
        disc = Discriminator_BIGAN(**disc_enc_network_args)   
        enc = Encoder_BIGAN(**disc_enc_network_args)  
        gen = Generator_BIGAN_GMM(model_args['optimize_prior'], model_args['num_clusters'], **gen_network_args)
    return {'disc': disc.to(device), 'enc': enc.to(device), 'gen': gen.to(device)}

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset_args = {'data': args.data, 
                    'data_dir': args.data_dir, 
                    'num_shots': -1,
                    'batch_size': args.batch_size,
                    'train': True, 
                    'normalize': True}
    train_loader, im_h, im_w, im_channels = setup_data_loader(**dataset_args)
    
    disc_enc_network_args = {'im_height': im_h, 
                             'im_width': im_w, 
                             'input_channels': im_channels, 
                             'channels': eval(args.channels), 
                             'kernels': eval(args.kernels), 
                             'strides': eval(args.strides), 
                             'paddings': eval(args.paddings),
                             'activation': args.activation,
                             'hidden_dim': eval(args.hidden_dim),
                             'latent_dim': args.latent_dim}
    
    gen_network_args = {'device': device,
                        'channels': eval(args.gen_channels), 
                        'kernels': eval(args.gen_kernels), 
                        'strides': eval(args.gen_strides), 
                        'paddings': eval(args.gen_paddings),
                        'activation': args.gen_activation,
                        'latent_dim': args.latent_dim}
    
    model_args = {'optimize_prior': args.optimize_prior,
                  'num_clusters': args.num_clusters}
    models = init_models(args.model_name, device, model_args, disc_enc_network_args, gen_network_args)
    exp_name = create_exp_name(args)
    print("Experiment: %s" % exp_name)

    trainer_args = {'models': models,
                    'train_loader': train_loader,
                    'num_epochs': args.num_epochs,
                    'device': device,
                    'exp_name': exp_name,
                    'optimizer': args.optimizer,
                    'lr': args.lr,
                    'image_noise_std': args.image_noise_std
                    }
    trainer = Train_BIGAN(**trainer_args)
    print('Start Training..')
    trainer.train()   

def parse_args():
    parser = argparse.ArgumentParser('BIGAN')
    parser.add_argument('--model_name', required=True, choices=['BIGAN', 'BIGAN_GMM'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--exp_id', default=None)
    ## data config
    parser.add_argument('--data', required=True, choices=['mnist', 'cifar10', 'svhn', 'fashionmnist'])    
    parser.add_argument('--data_dir', default='../datasets/', type=str)
    parser.add_argument('--image_noise_std', default=0.1, type=float)
    ## optim config
    parser.add_argument('--optimizer', choices=['AdamW', 'Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--optimize_prior', default=False, action='store_true')
    ## arch config
    parser.add_argument('--channels', default="[64,64,32,32]")
    parser.add_argument('--kernels', default="[3,4,4,4]")
    parser.add_argument('--strides', default="[1,2,2,2]")
    parser.add_argument('--paddings', default="[1,1,1,1]")
    parser.add_argument('--gen_kernels', default="[4,4,3,4,4]")
    parser.add_argument('--gen_channels', default="[64,64,32,32,1]") 
    parser.add_argument('--gen_strides', default="[1,2,2,2,2]")
    parser.add_argument('--gen_paddings', default="[1,1,1,1,1]")   
    parser.add_argument('--hidden_dim', default="[128]")
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--activation', default='LeakyReLU')
    parser.add_argument('--gen_activation', default='ReLU')
    parser.add_argument('--leak_slope', default=0.2, type=float)
    parser.add_argument('--dropout_prob', default=0.2, type=float)
    parser.add_argument('--num_clusters', default=20, type=int)
    ## training config
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    return parser.parse_args()
    
if __name__ == "__main__":    
    args = parse_args()
    main(args)