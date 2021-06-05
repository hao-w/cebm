#!/bin/sh

model=IGEBM_VERA

python train_vera.py --data=mnist \
                     --model_name=$model \
                     --device=cuda:0


python train_vera.py --data=fmnist \
                     --model_name=$model \
                     --device=cuda:0
                     
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
                     --lambda_ent=1e-4 \
                     --batch_size=64 \
                     --num_epochs=150 \
                     --device=cuda:0

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
                     --lambda_ent=1e-4 \
                     --batch_size=64 \
                     --num_epochs=15 \
                     --device=cuda:0