Source code for **Conjugate Energy-Based Models**([paper](https://openreview.net/forum?id=Asc_uGR8OkU)). 

### Abstract
In this paper, we propose conjugate energy-based models (CEBMs), a new class of energy-based models that define a joint density over data and latent variables. The joint density of a CEBM decomposes into an intractable distribution over data and a tractable posterior over latent variables. CEBMs have similar use cases as variational autoencoders, in the sense that they learn an unsupervised mapping from data to latent variables. However, these models omit a generator network, which allows them to learn more flexible notions of similarity between data points. Our experiments demonstrate that conjugate EBMs achieve competitive results in terms of image modelling, predictive power of latent space, and out-of-domain detection on a variety of datasets.


### Training Instructions
To train CEBM, CEBM_GMM, or the baseline IGEBM no a specific dataset, run the training python script with the following command:

```
python cebm/train/train_ebm.py --model_name=<the model name> --seed=<random_seed> --data=<the dataset name>
```


To train the baselines VAE or VAE_GMM, run the training python script with the following command:

  ```
  python cebm/train/train_vae.py --model_name=<the model name> --seed=<random_seed> --data=<the dataset name>
  ```

To train the baselines BIGAN or BIGAN_GMM, run the training python script with the following command:

  ```
  python cebm/train/train_bigan.py --model_name=<the model name> --seed=<random_seed> --data=<the dataset name>
  ```
