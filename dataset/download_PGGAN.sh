#!/bin/bash

wget -O PGGAN_128.tar --no-check-certificate https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/EdY_W5kqDlJPsCuMvbIKzrMBeBNJyBuDQ8F2Bo_-dxpbFw?download=1

tar -C ./PGGAN_128
tar -xvf PGGAN_128.tar

rm PGGAN_128.tar
