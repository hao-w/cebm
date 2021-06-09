#!/bin/sh

model=CEBM
device=cuda:1
sgld_steps=60
lr=1e-4

python3 train_ebm.py --data=mnist \
                     --model_name=$model \
                     --device=$device \
                     --sgld_steps=$sgld_steps \
                     --lr=$lr