#!/bin/bash

wget -O StyleGAN_256.tar --no-check-certificate https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/EZSOgqfry7lJiwcKwBmHH2MBPzqG7wNm-i7Lx5Wk94a3Ug?download=1

tar -C ./StyleGAN_256
tar -xvf StyleGAN_256.tar

rm StyleGAN_256.tar