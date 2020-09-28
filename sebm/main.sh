#!/bin/sh
# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=fashionmnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=100 --act=Swish --regularize_factor=5e-2 --arch=simplenet2

# sebm-svhn-standard
python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=cifar10 --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=1e-4 --channels=[64,128,256,512] --hidden_dim=[1024] --latent_dim=128 --num_epochs=100 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=60 --regularize_factor=1e-1 --activation=Swish --arch=simplenet2

# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=cifar10 --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --channels=[64,128,256,512] --hidden_dim=[1024] --latent_dim=128 --num_epochs=90 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=60 --regularize_factor=5e-3 --activation=Swish --arch=simplenet2

# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=svhn --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --channels=[64,128,256,512] --hidden_dim=[1024] --latent_dim=128 --num_epochs=90 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=60 --regularize_factor=5e-3 --activation=Swish --arch=simplenet2

#alonzo
# # clf-mnist
# python3 supervised_clf.py --seed=1 --device=1 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=200 --activation=ReLU

# python3 supervised_clf.py --seed=1 --device=1 --dataset=fashionmnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=200 --activation=ReLU
# 
# python3 supervised_clf.py --seed=1 --device=1 --dataset=cifar10 --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=150 --activation=ReLU
# 
# python3 supervised_clf.py --seed=1 --device=1 --dataset=svhn --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=200 --activation=ReLU

# # clf-heldon-size=1000
# python3 mlp_clf.py --seed=1 --device=1 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=10000 --activation=ReLU --heldon_size=1000

# python3 mlp_clf.py --seed=1 --device=1 --dataset=fashionmnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=10000 --activation=ReLU --heldon_size=1000

# python3 mlp_clf.py --seed=1 --device=1 --dataset=cifar10 --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=10000 --activation=ReLU --heldon_size=1000

# python3 mlp_clf.py --seed=1 --device=1 --dataset=svhn --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=1000 --activation=ReLU --heldon_size=1000

# # clf-heldon-size=100
# python3 mlp_clf.py --seed=1 --device=1 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=100000 --activation=ReLU --heldon_size=100

# python3 mlp_clf.py --seed=1 --device=1 --dataset=fashionmnist --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=100000 --activation=ReLU --heldon_size=100

# python3 mlp_clf.py --seed=1 --device=1 --dataset=cifar10 --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=100000 --activation=ReLU --heldon_size=100

# python3 mlp_clf.py --seed=1 --device=1 --dataset=svhn --data_dir=../../sebm_data/ --batch_size=100 --optimizer=Adam --lr=1e-4  --num_epochs=1000 --activation=ReLU --heldon_size=100



# clf-heldon-size=1000
# python3 mlp_clf.py --seed=1 --device=1 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=10 --optimizer=Adam --lr=1e-4  --num_epochs=5e+6 --activation=ReLU --heldon_size=10

# python3 mlp_clf.py --seed=1 --device=1 --dataset=fashionmnist --data_dir=../../sebm_data/ --batch_size=10 --optimizer=Adam --lr=1e-4  --num_epochs=5e+6  --activation=ReLU --heldon_size=10

# python3 mlp_clf.py --seed=1 --device=1 --dataset=cifar10 --data_dir=../../sebm_data/ --batch_size=10 --optimizer=Adam --lr=1e-4  --num_epochs=5e+6  --activation=ReLU --heldon_size=10

# python3 mlp_clf.py --seed=1 --device=1 --dataset=svhn --data_dir=../../sebm_data/ --batch_size=10 --optimizer=Adam --lr=1e-4  --num_epochs=10 --activation=ReLU --heldon_size=10
# vae-mnist-standard


# python3 vae.py --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-3  --latent_dim=2 --num_epochs=100 --arch=mlp --activation=ReLU

# sebm-Flowers102-standard
# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=flowers102 --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --op  timizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=900 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=60 --act=Swish --regularize_factor=5e-3 --arch=simplenet2





# python3 cebm_sgld.py --ss=2 --seed=1 --device=1 --dataset=cifar10 --data_dir=../../sebm_data/ --batch_size=256 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=200 --buffer_size=5000 --buffer_init --buffer_dup_allowed --sgld_noise_std=1e-2 --sgld_lr=2.0 --sgld_num_steps=20 --regularize_factor=5e-3 --warmup_iters=1000 --arch=wresnet --activation=LeakyReLU

# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=60 --act=Swish --regularize_factor=5e-3 --arch=simplenet2 

# sebm-FMNIST-standard


# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --act=Swish --regularize_factor=5e-3 --arch=simplenet --heldout_class=5

# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --act=Swish --regularize_factor=5e-3 --arch=simplenet --heldout_class=4
# 
# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --act=Swish --regularize_factor=5e-3 --arch=simplenet --heldout_class=9


# python3 cebm_sgld.py --ss=2 --seed=1 --device=0 --dataset=mnist --data_dir=../../sebm_data/ --batch_size=100 --data_noise_std=3e-2 --optimizer=Adam --lr=5e-5 --hidden_dim=[128] --latent_dim=128 --num_epochs=150 --buffer_init --buffer_dup_allowed --sgld_noise_std=7.5e-3 --sgld_lr=2.0 --sgld_num_steps=50 --act=Swish --regularize_factor=5e-3 --arch=simplenet --heldout_class=0



# python3 vae.py --seed=1 --device=0 --dataset=fashionmnist --data_dir=../../sebm_data/ --batch_size=100 --reparameterized --optimizer=Adam --lr=1e-4  --latent_dim=128 --num_epochs=1000 --arch=simplenet2 --activation=ReLU