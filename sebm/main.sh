#!/bin/sh
## CEBM training
# python3 cebm_sgld.py --seed=2 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=1e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init=True --buffer_dup_allowed=True --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --regularize_factor=1e-3


## EBM training 
# alonzo 0
# python3 ebm_sgld.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=1e-4 --hidden_dim=[128] --latent_dim=1 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --regularize_factor=1e-2
# # mlrg 1
python3 ebm_sgld.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=1e-4 --hidden_dim=[128] --latent_dim=1 --activation=LeakyReLU --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --regularize_factor=1e-2