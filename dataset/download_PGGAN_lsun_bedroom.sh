#!/bin/bash

wget -O PGGAN_lsun_bedroom.tar --no-check-certificate https://skku0-my.sharepoint.com/:u:/g/personal/byo7000_skku_edu/EZJoxQ2gjHVFl0uB0KFlWL0BxXqopW12-cZKnSsdlM4xXw?e=uULXa3

tar -C ./PGGAN_lsun_bedroom
tar -xvf PGGAN_lsun_bedroom.tar

rm PGGAN_lsun_bedroom.tar
