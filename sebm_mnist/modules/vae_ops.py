import torch
from quasi_conj.modules.encoder_vae import Encoder
from quasi_conj.modules.decoder_vae import Decoder

def init_modules(pixels_dim, hidden_dim, latents_dim, reparameterized, CUDA, DEVICE, LOAD_VERSION=None, LR=None):
    """
    initialize modules
    """
    encoder = Encoder(pixels_dim=pixels_dim, hidden_dim=hidden_dim, neural_ss_dim=latents_dim, reparameterized=reparameterized)
    decoder = Decoder(latents_dim=latents_dim, hidden_dim=hidden_dim, pixels_dim=pixels_dim, CUDA=CUDA, DEVICE=DEVICE)
    if CUDA:
        with torch.cuda.device(DEVICE):
            encoder.cuda()
            decoder.cuda()
    if LOAD_VERSION is not None:
        encoder.load_state_dict(torch.load('../weights/enc-%s' % LOAD_VERSION))
        decoder.load_state_dict(torch.load('../weights/dec-%s' % LOAD_VERSION))
    if LR is not None:
        assert isinstance(LR, float)
        optimizer =  torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=LR, betas=(0.9, 0.99))
        return (encoder, decoder), optimizer
    else:
        for p in encoder.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False    
        return (encoder, decoder)
    
def save_modules(modules, SAVE_VERSION):
    """
    ==========
    saving function
    ==========
    """
    (encoder, decoder) = modules
    torch.save(encoder.state_dict(), "../weights/enc-%s" % SAVE_VERSION)
    torch.save(decoder.state_dict(), "../weights/dec-%s" % SAVE_VERSION)
