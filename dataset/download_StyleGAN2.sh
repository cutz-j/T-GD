#!/bin/bash

wget -O StyleGAN2_256.tar --no-check-certificate https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/EV3nfdsCWSZGuE5uE0rFAuoBAqVms-Tn3s34rDn1UJFgjw?download=1

tar -C ./StyleGAN2_256
tar -xvf StyleGAN2_256.tar

rm StyleGAN2_256.tar
