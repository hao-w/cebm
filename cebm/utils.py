import os
from os.path import isfile
import numpy as np
import torch
import importlib
from datetime import datetime
import random

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
    # VAEs
    if args.model_name in ['VAE', 'VAE_GMM']:
        exp_name = '%s_d=%s_z=%s_lr=%s_samples=%s_seed=%d' % \
                    (args.model_name, args.data, args.latent_dim, args.lr, args.sample_size, args.seed)
        if args.model_name == 'VAE_GMM':
            exp_name += '_K=%d' % args.K
    # EBMs 
    elif args.model_name in ['IGEBM', 'CEBM', 'CEBM_GMM']:
        exp_name = '%s_d=%s_z=%s_lr=%s_act=%s_sgld_s=%s_n=%s_a=%s_dn=%s_reg=%s_seed=%d' % \
                    (args.model_name, args.data, args.latent_dim, args.lr, args.activation,
                     args.sgld_steps, args.sgld_noise_std, args.sgld_alpha, args.image_noise_std, 
                     args.regularize_coeff, args.seed)
        if args.model_name in ['CEBM', 'CEBM_GMM']:
            if args.optimize_ib:
                exp_name += '_learn-lam'
        if args.model_name == 'CEBM_GMM':
            exp_name += '_K=%d' % args.num_clusters
    # BIGANs
    elif args.model_name in ['BIGAN', 'BIGAN_GMM']:
        exp_name = '%s_d=%s_z=%s_lr=%s_dn=%s_seed=%d' % \
                    (args.model_name, args.data, args.latent_dim, args.lr, args.image_noise_std, args.seed)
        if args.model_name in ['BIGAN_GMM']:
            exp_name += '_K=%d' % args.num_clusters
    elif args.model_name in ['SEMI_CLF']:
        exp_name = '%s_d=%s_z=%s_lr=%s_act=%s_reg=%s_seed=%d' % \
                    (args.model_name, args.data, args.latent_dim, args.lr, args.activation, 
                     args.reg_lambda, args.seed)        
    else:
        raise ValueError()
    if args.exp_id is not None:
        exp_name += '_id=%s' % str(args.exp_id)
    return exp_name

def exp_finihsed(exp_name, epochs):
    "Checks if an experiment finished running"
    exp_foldername = os.path.join('saved/', exp_name)
    if not os.path.isdir(exp_foldername):
        return False
    else:
        checkpoints = [f for f in os.listdir(exp_foldername) if isfile(os.path.join(exp_foldername, f))]
        for ckp in checkpoints:
            if ('e=%d' % (epochs-1)) in ckp:
                return True
        return False

def delete_checkpoint(exp_name, e):
    exp_foldername = os.path.join('saved/', exp_name)
    checkpoint = exp_foldername + '/e=%d.rar' % e
    checkpoint_exist = os.path.isfile(checkpoint)
    if checkpoint_exist:
        os.remove(checkpoint)

def delete_previous_logs(exp_name):
    exp_foldername = os.path.join('saved/', exp_name, 'log')
    for filename in os.listdir(exp_foldername):
        if 'events' in filename:
            os.remove(os.path.join(exp_foldername, filename))

def save_checkpoint(trainer, exp_name, e):
    delete_checkpoint(exp_name, e-1) # delete previous checkpoint to avoid clutter
    exp_foldername = os.path.join('saved/', exp_name)
    if not os.path.isdir(exp_foldername):
        os.mkdir(exp_foldername)
    checkpoint_dict = {
    'model_state_dict': trainer.model.state_dict(),
    'trainer': trainer.state_dict()
    }
    torch.save(checkpoint_dict, exp_foldername + '/e=%d.rar' % e)

def load_checkpoint(exp_name, model, trainer, e):
    exp_foldername = os.path.join('saved/', exp_name)
    if not os.path.isdir(exp_foldername):
        print('experiment does not exists')
    else:
        checkpoint_dict = torch.load(exp_foldername + '/e=%d.rar' % e)
        checkpoint_dict['trainer']['config']['resume_checkpoint'] = e
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        trainer.load_state_dict(checkpoint_dict['trainer'])
    return model, trainer  
    
def linear(epoch, param_range, epoch_range):
    p_min, p_max = param_range
    e_min, e_max = epoch_range
    if (epoch < e_min):
        return p_min
    elif (epoch >= e_max):
        return p_max
    else:
        return np.linspace(p_min, p_max, e_max - e_min)[epoch - e_min]

#Hao added this new weight saving and loading functions
def save_models(models, filename, weights_dir="./weights"):
    checkpoint = {k: v.state_dict() for k, v in models.items()}
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    torch.save(checkpoint, f'{weights_dir}/{filename}')

def load_models(models, filename, weights_dir="./weights", **kwargs):
    checkpoint = torch.load(f'{weights_dir}/{filename}', **kwargs)
    {k: v.load_state_dict(checkpoint[k]) for k, v in models.items()}  
    
# From https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)

    h = (h_w[0] + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1

    return h, w

def deconv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of deconvolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)
    h = (h_w[0] - 1) * stride[0] - 2 * padding[0]  + (dilation * (kernel_size[0] - 1)) + 1
    w = (h_w[1] - 1) * stride[1] - 2 * padding[1]  + (dilation * (kernel_size[1] - 1)) + 1

    return h, w

def cnn_output_shape(h, w, kernels, strides, paddings):
    h_w = (h, w)
    for i, kernel in enumerate(kernels):
        h_w = conv_output_shape(h_w, kernels[i], strides[i], paddings[i])
    return h_w

def dcnn_output_shape(h, w, kernels, strides, paddings):
    h_w = (h, w)
    for i, kernel in enumerate(kernels):
        h_w = deconv_output_shape(h_w, kernels[i], strides[i], paddings[i])
    return h_w

def wres_block_params(stride, swap_cnn):
    kernels = [3,3]
    paddings = [1,1]
    if swap_cnn:
        strides = [1, stride]
    else:
        strides = [stride, 1]
    return kernels, strides, paddings
        

def conv_shape_print(h, w, kernels, strides, last_channel, pad=None):
    if pad is None:
        pad = [0 for i in kernels]
    print(h, w, "input")
    h_out = h
    w_out = w
    h_outs = []
    w_outs = []
    for i, kernel in enumerate(kernels):
        h_out, w_out = conv_output_shape((h_out, w_out), kernels[i], strides[i], pad[i])
        h_outs.append(h_out)
        w_outs.append(w_out)
        print(h_out, w_out, "(conv) out")

    print("###############")
    print("# neurons flatten: %d" % (last_channel * h_out * w_out))
    print("###############")
    h_out = h_outs[-1]
    w_out = w_outs[-1]
    for i, kernel in enumerate(kernels):
        h_out, w_out = deconv_output_shape((h_out, w_out), kernels[::-1][i], strides[::-1][i], pad[::-1][i])
        # if i == len(kernels) - 1:
        #     h_out, w_out = deconv_output_shape((h_out, w_out), kernels[::-1][i] + 2, strides[::-1][i], pad[::-1][i])
        # else:
        #     h_out, w_out = deconv_output_shape((h_out, w_out), kernels[::-1][i], strides[::-1][i], pad[::-1][i])
        print(h_out, w_out, "(deconv) out")
