#!/bin/sh

model=CEBM
device=cuda:0
likelihood=gaussian
sgld_steps=60

python3 train_ebm.py --data=mnist \
                     --model_name=$model \
                     --device=$device \
                     --sgld_steps=$sgld_steps

python3 train_ebm.py --data=fmnist \
                     --model_name=$model \
                     --device=$device \
                     --sgld_steps=$sgld_steps