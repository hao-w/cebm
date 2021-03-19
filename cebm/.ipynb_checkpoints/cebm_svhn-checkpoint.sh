#!/bin/sh

python3 cebm_sgld.py  --seed=1 --device=0 --dataset=svhn --data_dir=../../sebm_data/ --batch_size=100  \
--num_epochs=10 --buffer_init --buffer_dup_allowed  \
--hidden_dim=[128]