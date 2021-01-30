#!/bin/sh

python3 cebm_sgld.py --ss=2 --seed=123 --device=1 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=1e-2 --optimizer=Adam --lr=1e-4 --channels=[32,64,128,256,128] --strides=[1,2,2,2,2] --latent_dim=64 --num_epochs=100 --buffer_init --buffer_dup_allowed --sgld_noise_std=1e-2 --sgld_lr=2.0 --sgld_num_steps=50 --regularize_factor=0 --activation=LeakyReLU --leak=0.2 --arch=simplenet5
