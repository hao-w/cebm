#!/bin/sh

model=CEBM
device=cuda:0
likelihood=gaussian

python3 train_ebm.py --data=mnist \
                     --model_name=$model \
                     --device=$device

python3 train_ebm.py --data=fmnist \
                     --model_name=$model \
                     --device=$device