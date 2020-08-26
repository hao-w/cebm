#!/bin/sh
## CEBM training
# # mlrg
# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=cifar10 --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=1e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=200 --buffer_init --buffer_dup_allowed --sgld_noise_std=5e-3 --sgld_lr=20 --sgld_num_steps=60 --regularize_factor=1 --grad_clipping

# sebm-mnist-stanard
# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --act=Swish --regularize_factor=5e-3 --arch=simplenet2 

python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=flowers102 --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=900 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --act=Swish --regularize_factor=5e-3 --arch=simplenet2

# sebm-mnist-heldout
# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --act=Swish --regularize_factor=5e-3 --arch=simplenet2 --heldout_class=4

# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --act=Swish --regularize_factor=5e-3 --arch=simplenet2 --heldout_class=9

# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --act=Swish --regularize_factor=5e-3 --arch=simplenet2 --heldout_class=5

# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --act=Swish --regularize_factor=5e-3 --arch=simplenet2 --heldout_class=0


# vae-mnist-standard
# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-3  --latent_dim=128 --num_epochs=150 --arch=simplenet2 --activation=ReLU

# vae-mnist-heldout
# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-4  --latent_dim=128 --num_epochs=200 --arch=simplenet2 --activation=ReLU --heldout_class=0
# 
# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-4  --latent_dim=128 --num_epochs=200 --arch=simplenet2 --activation=ReLU --heldout_class=4

# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-4  --latent_dim=128 --num_epochs=200 --arch=simplenet2 --activation=ReLU --heldout_class=9

# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-4  --latent_dim=128 --num_epochs=200 --arch=simplenet2 --activation=ReLU --heldout_class=5

# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-4  --latent_dim=128 --num_epochs=150 --arch=simplenet2 --activation=ReLU --heldout_class=1

# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-4  --latent_dim=128 --num_epochs=150 --arch=simplenet2 --activation=ReLU --heldout_class=2

# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-4  --latent_dim=128 --num_epochs=150 --arch=simplenet2 --activation=ReLU --heldout_class=3

# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-4  --latent_dim=128 --num_epochs=150 --arch=simplenet2 --activation=ReLU --heldout_class=6


# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-4  --latent_dim=128 --num_epochs=150 --arch=simplenet2 --activation=ReLU --heldout_class=7

# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-4  --latent_dim=128 --num_epochs=150 --arch=simplenet2 --activation=ReLU --heldout_class=8


