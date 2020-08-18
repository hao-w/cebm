import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_samples(images_ebm):
    test_batch_size = len(images_ebm)
    images_ebm = images_ebm.squeeze().cpu().detach()
    images_ebm = torch.clamp(images_ebm, min=-1, max=1) * 0.5 + 0.5
    gs = gridspec.GridSpec(int(test_batch_size/10), 10)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(10, 10*int(test_batch_size/10)/ 10))
    for i in range(test_batch_size):
        ax = fig.add_subplot(gs[int(i/10), i%10])
        ax.imshow(images_ebm[i], cmap='gray', vmin=0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        
def set_seed(seed):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    
# From https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
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

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1

    return h, w

def deconv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
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

    if type(pad) is not tuple:
        pad = (pad, pad)
    h = (h_w[0] - 1) * stride[0] - 2 * pad[0]  + (dilation * (kernel_size[0] - 1)) + 1
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1]  + (dilation * (kernel_size[1] - 1)) + 1

    return h, w

def cnn_output_shape(h, w, kernels, strides, paddings=None):
    if paddings is None:
        paddings = [0 for i in kernels]
    h_in = h
    w_in = w
    for i, kernel in enumerate(kernels):
        h_in, w_in = conv_output_shape((h_in, w_in), kernels[i], strides[i], paddings[i])
    return h_in, w_in

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