import torch
import time
from torchvision import datasets, transforms

def train(optimizer, enc, dec, train_data, num_epochs, sample_size, batch_size, CUDA, DEVICE, SAVE_VERSION):
    """
    training the and encoder-decoder model using VAE objective
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
            trace = vae(enc, dec, images)
            (- trace['elbo']).backward()
            optimizer.step()
            for key in trace.keys():
                if key not in metrics:
                    metrics[key] = trace[key].detach()
                else:
                    metrics[key] += trace[key].detach()
        torch.save(enc.state_dict(), "../weights/rws-mlp-enc-%s" % SAVE_VERSION)
        torch.save(dec.state_dict(), "../weights/rws-mlp-dec-%s" % SAVE_VERSION)
        logging(metrics=metrics, filename=SAVE_VERSION, average_normalizer=b+1, epoch=epoch)
        time_end = time.time()
        print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, num_epochs, time_end - time_start))
        
def logging(metrics, filename, average_normalizer, epoch):
    if epoch == 0:
        log_file = open('../results/log-' + filename + '.txt', 'w+')
    else:
        log_file = open('../results/log-' + filename + '.txt', 'a+')
    metrics_print = ",  ".join(['%s=%.3e' % (k, v / average_normalizer) for k, v in metrics.items()])
    print("Epoch=%d, " % (epoch+1) + metrics_print, file=log_file)
    log_file.close()
    

def vae(enc, dec, images):
    """
    compute the ELBO in vae
    """
    trace = dict()
    latents, q_log_pdf = enc(images)
    p_log_pdf, recon, ll = dec(latents, images)
    log_w = (ll + p_log_pdf - q_log_pdf)
    trace['elbo'] = log_w.mean()
    return trace 
    
if __name__ == "__main__":
    import torch
    from basics.nets.mlp_encoder import Encoder
    from basics.nets.mlp_decoder import Decoder
    CUDA = torch.cuda.is_available()
    if CUDA:
        DEVICE = torch.device('cuda:0')
    print('torch:', torch.__version__, 'CUDA:', CUDA)
    num_epochs = 150
    batch_size = 100
    pixel_dim = 784
    sample_size = 1
    hidden_dim = 128
    latent_dim = 10
    lr = 1 * 1e-4
    reparameterized = True
    ## EBM hyper-parameters
    SAVE_VERSION = 'vae-mnist-zd=%s' % (latent_dim) 
    
    ## data directory
    print('Load MNIST dataset...')
    data_dir = '/home/hao/Research/sebm_data/'
    transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,),(0.5,))]) 
    data = torch.utils.data.DataLoader(
                        datasets.MNIST(data_dir, train=train, download=True,
                                       transform=transform),
                        batch_size=batch_size, shuffle=True) 
    
    print('Initialize encoder and decoder and optimizer...')
    enc = Encoder(latent_dim, hidden_dim, pixel_dim, reparameterized=reparameterized)
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
          train_data=data, 
          num_epochs=num_epochs, 
          sample_size=sample_size, 
          batch_size=batch_size,
          CUDA=CUDA, 
          DEVICE=DEVICE, 
          SAVE_VERSION=SAVE_VERSION)
    
    
