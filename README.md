# T-GD: Transferable GAN-generated Images Detection Framework.

## Overview of our framework.
<img src='./image/overview.png' width=1200>

## Dataset
The dataset for each result condition can be downloaded by running the file in dataset or [here](https://skku0-my.sharepoint.com/:f:/g/personal/byo7000_skku_edu/EoP8mWpbyDhNtIaZ9rBoPWcB5QRsinPBKwr0V18dHsUR8w?e=7oNCXY).

A example script for downloading the testset is as follows:

```
# Download the dataset
cd dataset
bash download_StarGAN.sh
bash download_StyleGAN.sh
bash download_StyleGAN2.sh
cd ..
```

For the PGGAN dataset, we have contact with the original author about opening the dataset, and waiting for a reply. After respond we will upload the PGGAN datset in future.

## Download pre-trained model weights
The pretrained weights can be downloaded by running the file in dataset or [here](https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/ESF_WuwUau5Mrl3hdteSChcBX3mBt8SP1Ajq-CmkEN7MlQ?e=SeLff1).

```
# Download the pre-trained weights
cd pre-trained_weights
bash download_weights.sh
cd ..
```
