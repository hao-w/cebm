import torch.nn.functional as F

def rws(enc, dec, images):
    """
    compute the EUBO in rws and the separate gradient w.r.t. phi and theta
    """
    trace = dict()
    latents, q_log_pdf = enc(images)
    p_log_pdf, recon, ll = dec(latents, images)
    log_w = (ll + p_log_pdf - q_log_pdf).detach()
    w = F.softmax(log_w, 0)
    trace['loss_theta'] = (- w * ll).sum(0).mean()
    trace['loss_phi'] = (- w * q_log_pdf).sum(0).mean()
    trace['eubo'] = (w * log_w).sum(0).mean()
    trace['elbo'] = log_w.mean()
    trace['ess'] = (1 / (w**2).sum(0)).mean()
    return trace 

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

def ae(enc, dec, images):
    """
    compute the ELBO in vae
    """
    trace = dict()
    latents = enc(images)
    recon, ll = dec(latents, images)
    trace['loss'] = - ll.mean()
    return trace 

    