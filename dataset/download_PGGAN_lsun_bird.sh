#!/bin/bash

wget -O PGGAN_lsun_bird.tar --no-check-certificate https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/EaF-Pc6kiiRPs8vIsw2MY-kBSIj-ZRbg1bqHX7fDn2vIsA?download=1

tar -C ./PGGAN_lsun_bird
tar -xvf PGGAN_lsun_bird.tar

rm PGGAN_lsun_bird.tar
