#!/bin/bash

wget -O StyleGAN2_256.tar --no-check-certificate https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/EZSOgqfry7lJiwcKwBmHH2MBPzqG7wNm-i7Lx5Wk94a3Ug?download=1

tar -C ./StyleGAN2_256
tar -xvf StyleGAN2_256.tar

rm StyleGAN2_256.tar
