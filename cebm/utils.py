import os
import torch
import random
import numpy as np
from cebm.model.ebm import CEBM_Gaussian, CEBM_GMM, IGEBM
from cebm.model.vae import Encoder_VAE, Decoder_VAE_Gaussian, Decoder_VAE_GMM
from cebm.model.gan import Discriminator_BIGAN, Encoder_BIGAN, Generator_BIGAN, Generator_BIGAN_GMM

def set_seed(seed):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

def create_exp_name(args):
    # VAEs and BIGANs
    if args.model_name in ['VAE', 'VAE_GMM', 'BIGAN', 'BIGAN_GMM']:
        exp_name = '%s_d=%s_z=%s_lr=%s_seed=%d' % \
                    (args.model_name, args.data, args.latent_dim, args.lr, args.seed)
        if args.model_name in ['VAE_GMM', 'BIGAN_GMM']:
            exp_name += '_K=%d' % args.num_clusters
        if args.optimize_prior:
            exp_name+= '_learn-prior'
    # EBMs 
    elif args.model_name in ['IGEBM', 'CEBM', 'CEBM_GMM']:
        exp_name = '%s_d=%s_z=%s_lr=%s_act=%s_sgld_s=%s_n=%s_a=%s_dn=%s_reg=%s_seed=%d' % \
                    (args.model_name, args.data, args.latent_dim, args.lr, args.activation,
                     args.sgld_steps, args.sgld_noise_std, args.sgld_alpha, args.image_noise_std, 
                     args.regularize_coeff, args.seed)
        if args.model_name == 'CEBM_GMM':
            exp_name += '_K=%d' % args.num_clusters
        if args.optimize_ib:
            exp_name += '_learn-prior'
            
    elif args.model_name == 'SEMI_CLF':
        exp_name = '%s_d=%s_z=%s_lr=%s_act=%s_reg=%s_seed=%d' % \
                    (args.model_name, args.data, args.latent_dim, args.lr, 
                     args.reg_lambda, args.seed)  
        
    elif args.model_name == 'META_GMVAE':
        exp_name = 'META_GMVAE_d=%s_z=%s' % (args.data, args.latent_dim)
    else:
        raise ValueError
        
    if args.exp_id is not None:
        exp_name += '_id=%s' % str(args.exp_id)
        
    print("Experiment: %s" % exp_name)
    return exp_name

def init_models(model_name, device, model_args, network_args):
#     if data in ['mnist', 'fashionmnist']:
#         network_args = {'device': device,
#                         'im_height': im_h, 
#                         'im_width': im_w, 
#                         'input_channels': im_channels, 
#                         'channels': [64,64,32,32], 
#                         'kernels': [3,4,4,4], 
#                         'strides': [1,2,2,2], 
#                         'paddings': [1,1,1,1],
#                         'hidden_dims': [128],
#                         'latent_dim': latent_dim,
#                         'activation': activation,
#                         'leaky_slope': 0.2}
        
#         if model_name in ['VAE', 'VAE_GMM']:
#             network_args['dec_paddings'] = [1,1,0,0]
#     else:
#         network_args = {'device': device,
#                         'im_height': im_h, 
#                         'im_width': im_w, 
#                         'input_channels': im_channels, 
#                         'channels': [64,128,256,512], 
#                         'kernels': [3,4,4,4], 
#                         'strides': [1,2,2,2], 
#                         'paddings': [1,1,1,1],
#                         'hidden_dims': [1024,256],
#                         'latent_dim': latent_dim,
#                         'activation': activation,
#                         'leaky_slope': 0.2}  
#         if model_name in ['VAE', 'VAE_GMM']:
#             raise ValueError('need to specify the dec_paddings for decoder')
# #             network_args['dec_paddings']
        
    if model_name in ['CEBM', 'CEBM_GMM', 'IGEBM']:
        if model_name == 'CEBM':
            model = CEBM_Gaussian(**network_args)        
        elif model_name == 'CEBM_GMM':
            model = CEBM_GMM(model_args['optimize_ib'], model_args['num_clusters'], **network_args)   
        elif model_name == 'IGEBM':
            model = IGEBM(**network_args)
        return {'ebm': model.to(device)}
    
    elif model_name in ['VAE', 'VAE_GMM']:
        enc = Encoder_VAE(**network_args) 
        if model_name == 'VAE':
            dec = Decoder_VAE_Gaussian(**network_args)
        elif model_name == 'VAE_GMM':
            dec = Decoder_VAE_GMM(model_args['optimize_prior'], model_args['num_clusters'], **network_args)
        return {'enc': enc.to(device), 'dec': dec.to(device)}

    elif model_name in ['BIGAN', 'BIGAN_GMM']:
        disc = Discriminator_BIGAN(**network_args)   
        enc = Encoder_BIGAN(**network_args)  
        if model_name == 'BIGAN':
            gen = Generator_BIGAN(**network_args)
        elif model_name == 'BIGAN_GMM': 
            gen = Generator_BIGAN_GMM(model_args['optimize_prior'], model_args['num_clusters'], **network_args)
        return {'disc': disc.to(device), 'enc': enc.to(device), 'gen': gen.to(device)}

def save_models(models, filename, weights_dir="./weights"):
    checkpoint = {k: v.state_dict() for k, v in models.items()}
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    torch.save(checkpoint, f'{weights_dir}/{filename}')

def load_models(models, filename, weights_dir="./weights", **kwargs):
    checkpoint = torch.load(f'{weights_dir}/{filename}', **kwargs)
    {k: v.load_state_dict(checkpoint[k]) for k, v in models.items()}  
    

