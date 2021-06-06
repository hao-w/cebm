#!/bin/sh

model=CEBM_GMM_VERA
device=cuda:1
likelihood=cb
lambda_ent=1e-4

python train_vera.py --data=mnist \
                     --model_name=$model \
                     --device=$device \
                     --lr_xee=1e-3 \
                     --likelihood=$likelihood

python train_vera.py --data=fmnist \
                     --model_name=$model \
                     --device=$device \
                     --lr_xee=1e-3 \
                     --likelihood=$likelihood
                     
python train_vera.py --data=cifar10 \
                     --model_name=$model \
                     --channels=[64,128,256,512] \
                     --kernels=[3,4,4,4] \
                     --strides=[1,2,2,2] \
                     --paddings=[1,1,1,1] \
                     --hidden_dims=[1024] \
                     --gen_kernels=[4,4,4,4,4] \
                     --gen_channels=[512,256,128,64,3] \
                     --gen_strides=[2,2,2,2,2] \
                     --gen_paddings=[1,1,1,1,1] \
                     --lambda_ent=$lambda_ent \
                     --batch_size=64 \
                     --num_epochs=200 \
                     --device=$device \
                     --likelihood=$likelihood

python train_vera.py --data=svhn \
                     --model_name=$model \
                     --channels=[64,128,256,512] \
                     --kernels=[3,4,4,4] \
                     --strides=[1,2,2,2] \
                     --paddings=[1,1,1,1] \
                     --hidden_dims=[1024] \
                     --gen_kernels=[4,4,4,4,4] \
                     --gen_channels=[512,256,128,64,3] \
                     --gen_strides=[2,2,2,2,2] \
                     --gen_paddings=[1,1,1,1,1] \
                     --lambda_ent=$lambda_ent \
                     --batch_size=64 \
                     --num_epochs=15 \
                     --device=$device \
                     --likelihood=$likelihood