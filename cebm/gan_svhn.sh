#!/bin/sh



# python3 dcgan.py --seed=1 --device=0 --dataset=svhn --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=2e-4  --num_epochs=40 \
# --disc_channels=[64,128,256,512] --disc_kernels=[3,4,4,4] --disc_strides=[1,2,2,2] --disc_paddings=[1,1,1,1] --hidden_dim=[128,128] \
# --gen_channels=[512,256,128,64,3] --gen_kernels=[4,4,4,4,4] --gen_strides=[2,2,2,2,2] --gen_paddings=[1,1,1,1,1]

# python3 dcgan_gmm.py --seed=1 --device=0 --dataset=svhn --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=2e-4  --num_epochs=15 \
# --disc_channels=[64,128,256,512] --disc_kernels=[3,4,4,4] --disc_strides=[1,2,2,2] --disc_paddings=[1,1,1,1] --hidden_dim=[128,128] \
# --gen_channels=[512,256,128,64,3] --gen_kernels=[4,4,4,4,4] --gen_strides=[2,2,2,2,2] --gen_paddings=[1,1,1,1,1] 


python3 bigan.py --seed=1 --device=1 --dataset=svhn --data_dir=../../sebm_data/ \
--batch_size=100 --optimizer=Adam --lr=2e-4  --num_epochs=10 \
--disc_channels=[64,128,256,512] --disc_kernels=[3,4,4,4] --disc_strides=[1,2,2,2] --disc_paddings=[1,1,1,1] \
--hidden_dim=[1024] --cnn_output_dim=8192 \
--gen_channels=[512,256,128,64,3] --gen_kernels=[4,4,4,4,4] --gen_strides=[2,2,2,2,2] --gen_paddings=[1,1,1,1,1] \
--enc_channels=[64,128,256,512] --enc_kernels=[3,4,4,4] --enc_strides=[1,2,2,2] --enc_paddings=[1,1,1,1]


# python3 bigan_gmm.py --seed=1 --device=0 --dataset=svhn --data_dir=../../sebm_data/ --batch_size=100 \
# --optimizer=Adam --lr=2e-4  --num_epochs=10 \
# --disc_channels=[64,128,256,512] --disc_kernels=[3,4,4,4] --disc_strides=[1,2,2,2] \
# --disc_paddings=[1,1,1,1]  --hidden_dim=[1024] --cnn_output_dim=8192 \
# --gen_channels=[512,256,128,64,3] --gen_kernels=[4,4,4,4,4] \
# --gen_strides=[2,2,2,2,2] --gen_paddings=[1,1,1,1,1] \
# --enc_channels=[64,128,256,512] --enc_kernels=[3,4,4,4] --enc_strides=[1,2,2,2] --enc_paddings=[1,1,1,1]