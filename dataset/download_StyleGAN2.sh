#!/bin/bash

wget -O StyleGAN2_256.tar.001 --no-check-certificate https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/EQCPywddQ8VLvB5jU_l8NIABCv2Dwv77-5AYLmEa9oRyig?download=1

wget -O StyleGAN2_256.tar.002 --no-check-certificate https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/Ee5Dco4narZFigMyCFnqrc0BnGmNr2E8OH81RbLrlsiQIw?download=1

tar -C ./StyleGAN2_256
cat StyleGAN2_256.tar* | tar xvf -

rm StyleGAN2_256.tar.001
rm StyleGAN2_256.tar.002
