# T-GD: Transferable GAN-generated Images Detection Framework.
Hyeonseong Jeon, Youngoh Bang, Junyaup Kim, and Simon S. Woo. "T-GD: Transferable GAN-generated Images Detection Framework.". Thirty-seventh International Conference on Machine Learning  (ICML). 2020.

## Overview of our framework.
<img src='./image/overview.png' width=1000>

## Clone
```
git clone https://github.com/cutz-j/T-GD
```

## Dataset
The dataset for each result condition can be downloaded by running the file in dataset or [here](https://skku0-my.sharepoint.com/:f:/g/personal/byo7000_skku_edu/EoP8mWpbyDhNtIaZ9rBoPWcB5QRsinPBKwr0V18dHsUR8w?e=7oNCXY).

CelebA, CelebA-HQ, FFHQ, LSUN-Bedroom, LSUN-Bird.

PGGAN_CelebA, StarGAN, StyleGAN1, StyleGAN2, PGGAN_Bedroom, PGGAN_Bird.

A example script for downloading the testset is as follows:

```
# Download the dataset
cd dataset
bash download_PGGAN.sh
bash download_StarGAN.sh
bash download_StyleGAN.sh
bash download_StyleGAN2.sh
bash download_PGGAN_lsun_bedroom.sh
bash download_PGGAN_lsun_bird.sh
cd ..
```

For the PGGAN dataset, we have contacted with the dataset provider about opening the dataset. For now we only uploaded data for the test set.
PGGAN-images from LSUN-bedroom and LSUN-bird will be added to dataset.

Our datasets are heavily from [here](https://github.com/tkarras/progressive_growing_of_gans).

## Download pre-trained model weights
The pretrained weights can be downloaded by running the file in dataset or [here](https://skku0-my.sharepoint.com/:f:/g/personal/byo7000_skku_edu/EoP8mWpbyDhNtIaZ9rBoPWcB5QRsinPBKwr0V18dHsUR8w?e=7oNCXY).

```
# Download the pre-trained weights
cd weights
bash download_weights.sh
cd ..
```

## Setup
```
pip install -r requirements.txt
```
* warmup scheduler
```
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```

## Evaluation
```
# Dataset and model weights need to be downloaded.
# source and target dataset dir. i.e., StarGAN --> StyleGAN2
# pretrained weight. i.e., efficientnet/stargan.pth.tar
# t-gd pretrained weight. i.e., t-gd/efficientnet/star_to_style2.pth.tar
python eval.py --source_dataset dataset/StarGAN_128 \
                --target_dataset dataset/StyleGAN2_256 \
                --pretrained_dir weights/pre-train/efficientnet/stargan.pth.tar \
                --resume weights/t-gd/efficientnet/star_to_style2.pth.tar
```

## Pre-train
```
# Dataset needs to be downloaded.
# source dataset

python pretrain.py --source_dataset dataset/StarGAN_128

```

## Transfer
```
# Dataset and model weights are needed.
# source and target dataset dir. i.e., StarGAN --> StyleGAN2
# pretrained weight. i.e., efficientnet/stargan.pth.tar

python transfer.py --target style2
                --source_dataset dataset/StarGAN_128 \
                --target_dataset dataset/StyleGAN2_256 \
                --pretrained_dir weights/pre-train/efficientnet/stargan.pth.tar
```
