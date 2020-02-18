import os
import shutil
import numpy as np

source_dir = 'z:/media/data2/dataset/GAN_ImageData/StyleGAN_64/train/'
target_dir = 'z:/media/data2/dataset/GAN_ImageData/StyleGAN_64/validation/'

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)
    os.mkdir(os.path.join(target_dir, '0'))
    os.mkdir(os.path.join(target_dir, '1'))    

real = os.listdir(os.path.join(source_dir, '0'))
fake = os.listdir(os.path.join(source_dir, '1'))

sample_size = len(real)//10

real_sample = np.random.choice(real, size=sample_size, replace=False)
fake_sample = np.random.choice(fake, size=sample_size, replace=False)

for i in real_sample:
    shutil.move(os.path.join(source_dir+'0', i), os.path.join(target_dir+'0', i))
    
for j in fake_sample:
    shutil.move(os.path.join(source_dir+'1', j), os.path.join(target_dir+'1', j))