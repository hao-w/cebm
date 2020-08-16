import torch
import time
from ffebm.objectives import ae

def train(optimizer, enc, dec, train_data, num_epochs, batch_size, CUDA, DEVICE, SAVE_VERSION, logging_freq):
    """
    training the and autoencoder using cnn architecture
    """
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = dict()
        for b, images in enumerate(train_data):
            batch_size, _, pixel_size_sqrt, _ = images.shape
            optimizer.zero_grad()
            if CUDA:
                images = images.cuda().to(DEVICE)
            trace = ae(enc, dec, images)
            (trace['loss']).backward()
            optimizer.step()
            for key in trace.keys():
                if key not in metrics:
                    metrics[key] = trace[key].detach()
                else:
                    metrics[key] += trace[key].detach()
            if (b+1) % logging_freq == 0:
                logging(metrics=metrics, filename=SAVE_VERSION, average_normalizer=b+1, epoch=epoch, logging_freq=logging_freq)
                metrics = dict()
                torch.save(enc.state_dict(), "../weights/rws-mlp-enc-%s" % SAVE_VERSION)
                torch.save(dec.state_dict(), "../weights/rws-mlp-dec-%s" % SAVE_VERSION)
        
                time_end = time.time()
                print("Epoch=%d / %d, GS=%d completed  in (%ds),  " % (epoch+1, num_epochs, b+1, time_end - time_start))
                time_start = time.time()
        
def logging(metrics, filename, average_normalizer, epoch, logging_freq):
    if epoch == 0 and average_normalizer == logging_freq:
        log_file = open('../results/log-' + filename + '.txt', 'w+')
    else:
        log_file = open('../results/log-' + filename + '.txt', 'a+')
    metrics_print = ",  ".join(['%s=%.3e' % (k, v / logging_freq) for k, v in metrics.items()])
    print("Epoch=%d, GS=%d, " % (epoch+1, average_normalizer) + metrics_print, file=log_file)
    log_file.close()
    

    
if __name__ == "__main__":
    import torch
    from ffebm.patches import load_mnist_patches
    from ffebm.nets.cnn_autoencoder import Encoder
    from ffebm.nets.mlp_autodecoder import Decoder
    CUDA = torch.cuda.is_available()
    if CUDA:
        DEVICE = torch.device('cuda:0')
    print('torch:', torch.__version__, 'CUDA:', CUDA)
    num_epochs = 1000
    batch_size = 100
    pixel_dim = 9
    latent_dim = 32
    hidden_dim = 128
    lr = 1 * 1e-4
    ## EBM hyper-parameters
    SAVE_VERSION = 'mnistpatch-ae-%.2Elr-%.2Elatentdim' % (lr, latent_dim) 
    
    ## data directory
    patch_size = 3
    logging_freq = 10000
    print('Load MNIST patches dataset...')
    mnist_data_dir = '../../../sebm_data/'
    train_path = mnist_data_dir+'MNIST_patch/train_data_%d_by_%d.pt' % (patch_size, patch_size)
    test_path = mnist_data_dir+'MNIST_patch/test_data_%d_by_%d.pt' % (patch_size, patch_size)
    patch_data_dir = (train_path, test_path)
    
    train_data, test_data = load_mnist_patches(patch_data_dir, batch_size)
    
    print('Initialize encoder and decoder and optimizer...')
    enc = Encoder(latent_dim)
    dec = Decoder(latent_dim, hidden_dim, pixel_dim)
#     print('Initialize proposal...')
#     proposal = Proposal(latent_dim=latent_dim)
    if CUDA:
        with torch.cuda.device(DEVICE):
            enc.cuda()  
            dec.cuda()
    optimizer = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=lr, betas=(0.9, 0.999))

    print('Start training...')
    train(optimizer=optimizer, 
          enc=enc, 
          dec=dec,
          train_data=train_data, 
          num_epochs=num_epochs, 
          batch_size=batch_size,
          CUDA=CUDA, 
          DEVICE=DEVICE, 
          SAVE_VERSION=SAVE_VERSION,
          logging_freq=logging_freq)
    
    
