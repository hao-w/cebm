#!/bin/sh

model=CEBM_GMM
device=cuda:0
sgld_steps=60
lr=5e-5
num_clusters=20

python3 train_ebm.py --data=cifar10 \
                     --model_name=$model \
                     --channels=[64,128,256,512] \
                     --kernels=[3,4,4,4] \
                     --strides=[1,2,2,2] \
                     --paddings=[1,1,1,1] \
                     --hidden_dims=[1024] \
                     --device=$device \
                     --sgld_steps=$sgld_steps \
                     --lr=$lr \
                     --num_clusters=$num_clusters \
                     --optimize_ib