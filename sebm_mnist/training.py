import torch
import time
from quasi_conj.modules.module_ops import save_modules
def train(objective, opt_schedule, optimizer_phi, optimizer_theta, modules, train_data, num_epochs, sample_size, batch_size, CUDA, DEVICE, SAVE_VERSION):
    """
    training function with KL objective
    """
    (enc, dec) = modules
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = {'loss_phi' : 0.0, 'loss_theta' : 0.0, 'unnormalized' : 0.0, 'normalizing_constant' : 0.0}
        for b, (images, _) in enumerate(train_data):
            optimizer_phi.zero_grad()    
            optimizer_theta.zero_grad()    
#             optimizer.zero_grad()
            images = images.squeeze(1).view(-1, 28*28).repeat(sample_size, 1, 1)
            if CUDA:
                images = images.cuda().to(DEVICE)
            loss_phi, loss_theta, unnormalized, normalizing_constant = objective(modules=modules, images=images)
            if opt_schedule is None:
                (loss_phi+loss_theta).backward()
                optimizer_phi.step()
                optimizer_theta.step()
            else:
                if b % opt_schedule == 0:
#                     print('optimize phi')
                    loss_phi.backward()
                    optimizer_phi.step()
                else:
                    loss_theta.backward()
                    optimizer_theta.step()
#                     print('optimize theta')
            metrics['unnormalized'] += unnormalized.detach()
            metrics['normalizing_constant'] += normalizing_constant.detach()
            metrics['loss_theta'] += loss_theta.detach()
            metrics['loss_phi'] += loss_phi.detach()
#             metrics['ess'] += ess
        save_modules(modules=modules, SAVE_VERSION=SAVE_VERSION)
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