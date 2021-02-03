#!/bin/sh

python3 dcgan.py --seed=1 --device=1 --dataset=fashionmnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=2e-4  --num_epochs=150 

python3 dcgan_gmm.py --seed=1 --device=1 --dataset=fashionmnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=2e-4  --num_epochs=150 

# python3 bigan_gmm.py --seed=1 --device=1 --dataset=fashionmnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=2e-4  --num_epochs=150 
