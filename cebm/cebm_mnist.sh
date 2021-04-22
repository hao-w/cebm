#!/bin/sh

python3 cebm_gmm_sgld.py  --seed=1 --device=1 --dataset=mnist --data_dir=../../sebm_data/ \
--batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=1e-4  --num_epochs=100 \
--buffer_init --buffer_dup_allowed  --optimize_priors

python3 cebm_gmm_sgld.py --seed=1 --device=1 --dataset=fashionmnist --data_dir=../../sebm_data/ \
--num_epochs=100 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 \
--optimize_priors