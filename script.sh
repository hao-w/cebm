#!/bin/sh

python cebm/main.py --train --eval --seed=0 --device=1 --data=mnist --lr=1e-3 --optim=Adam --model=GMM_VAE --net=cnn --activation=ReLU --num-latent=128  --channels=[32,32,64,64] --kernels=[4,4,4,4] --strides=[2,2,2,2] --paddings=[3,1,1,1] --num-neurons=[128] --epochs=10

# python cebm/main.py --load --eval --seed=0 --device=1 --exp-id=test --data=mnist --lr=1e-3 --model=VAE --net=cnn --num-latent=128  --channels=[32,32,64,64] --kernels=[4,4,4,4] --strides=[2,2,2,2] --paddings=[3,1,1,1] --num-neurons=[128] --epochs=20
