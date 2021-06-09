#!/bin/sh

model=CEBM
device=cuda:1
likelihood=gaussian
sgld_steps=60
lr=5e-5

python3 train_ebm.py --data=mnist \
                     --model_name=$model \
                     --device=$device \
                     --sgld_steps=$sgld_steps \
                     --lr=$lr