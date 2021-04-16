#!/bin/sh

# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=cifar10 --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=1e-2 --optimizer=Adam --lr=5e-5 --num_epochs=200 --buffer_init --buffer_dup_allowed --sgld_noise_std=1e-3 --sgld_lr=2.0 --sgld_num_steps=40 --regularize_factor=1e-2 --activation=Swish --leak=0.1 --arch=simplenet2 --dropout=0.2


# python3 cebm_sgld.py  --seed=1 --device=0 --dataset=cifar10 --data_dir=../../sebm_data/ --batch_size=100  \
# --num_epochs=100 --buffer_init --buffer_dup_allowed  \
# --hidden_dim=[128]




python3 cebm_joint_sgld.py  --seed=1 --device=0 --dataset=cifar10 --num_epochs=150 --buffer_init --buffer_dup_allowed 

python3 cebm_joint_sgld.py  --seed=1 --device=1 --dataset=svhn --num_epochs=15 --buffer_init --buffer_dup_allowed 


# python3 cebm_gmm_sgld.py  --seed=1 --device=0 --dataset=cifar10 --num_epochs=150 --buffer_init --buffer_dup_allowed 

# python3 cebm_gmm_sgld.py  --seed=1 --device=1 --dataset=svhn --num_epochs=15 --buffer_init --buffer_dup_allowed 
