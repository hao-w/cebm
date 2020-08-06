import torch
import time
from ffebm.objectives import rws

def train(optimizer, enc, dec, train_data, num_epochs, sample_size, batch_size, CUDA, DEVICE, SAVE_VERSION):
    """
    training the and encoder-decoder model using RWS
    """
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = dict()
        for b, (images, _) in enumerate(train_data):
            batch_size, _, pixel_size_sqrt, _ = images.shape
            images = images.squeeze(1).view(batch_size, pixel_size_sqrt*pixel_size_sqrt).repeat(sample_size, 1, 1)
            optimizer.zero_grad()
            if CUDA:
                images = images.cuda().to(DEVICE)
            trace = rws(enc, dec, images)
            (trace['loss_theta'] + trace['loss_phi']).backward()
            optimizer.step()
        torch.save(ef.state_dict(), "../weights/ef-%s" % SAVE_VERSION)
        for key in trace.keys():
            if key not in metrics.keys():
                metrics[key] = trace[key].detach()
            else:
                metrics[key] += trace[key].detach()
        logging(metrics=metrics, filename=SAVE_VERSION, average_normalizer=b+1, epoch=epoch)
        time_end = time.time()
        print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, num_epochs, time_end - time_start))
        
def logging(metrics, filename, average_normalizer, epoch):
    if epoch == 0:
        log_file = open('../results/log-' + filename + '.txt', 'w+')
    else:
        log_file = open('../results/log-' + filename + '.txt', 'a+')
    metrics_print = ",  ".join(['%s=%.5e' % (k, v / average_normalizer) for k, v in metrics.items()])
    print("Epoch=%d, " % (epoch+1) + metrics_print, file=log_file)
    log_file.close()
    

    
if __name__ == "__main__":
    import torch
    from ffebm.data import load_mnist
    from ffebm.nets.mlp_encoder import Encoder
    from ffebm.nets.mlp_decoder import Decoder
    CUDA = torch.cuda.is_available()
    if CUDA:
        DEVICE = torch.device('cuda:0')
    print('torch:', torch.__version__, 'CUDA:', CUDA)
    num_epochs = 1000
    batch_size = 100
    pixel_dim = 784
    sample_size = 10
    hidden_dim = 128
    latent_dim = 10
    lr = 1 * 1e-4
    ## EBM hyper-parameters
    SAVE_VERSION = 'mnist-rws-%.2Elr-%.2Elatentdim' % (lr, latent_dim) 
    
    ## data directory
    print('Load MNIST dataset...')
    DATA_DIR = '/home/hao/Research/sebm_data/'
    train_data, test_data = load_mnist(DATA_DIR, batch_size, resize=None)
    
    print('Initialize encoder and decoder and optimizer...')
    enc = Encoder(latent_dim, hidden_dim, pixel_dim)
    dec = Decoder(latent_dim, hidden_dim, pixel_dim, CUDA, DEVICE)
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
          sample_size=sample_size, 
          batch_size=batch_size,
          CUDA=CUDA, 
          DEVICE=DEVICE, 
          SAVE_VERSION=SAVE_VERSION)
    
    
