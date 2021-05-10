#!/bin/bash

wget -O StyleGAN_256.tar --no-check-certificate https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/EV3nfdsCWSZGuE5uE0rFAuoBAqVms-Tn3s34rDn1UJFgjw?download=1

tar -C ./StyleGAN_256
tar -xvf StyleGAN_256.tar

rm StyleGAN_256.tar
