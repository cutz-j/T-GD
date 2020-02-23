import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='efficientnet-b0', help='architecture for binary classification')
        parser.add_argument('--checkpoint', default='./log/')

        # data augmentation
        parser.add_argument('--classes', default=2)
        parser.add_argument('--epochs', default=400)
        parser.add_argument('--iterations', default=1000)
        parser.add_argument('--start_epoch', default=0)
        parser.add_argument('--train_batch', default=200)
        parser.add_argument('--test_batch', default=200)
        parser.add_argument('--lr', default=0.04)
        parser.add_argument('--schedule', default=[50, 250, 500, 750])
        parser.add_argument('--momentum', default=0.1)
        parser.add_argument('--gamma', default=0.1)

        parser.add_argument('--num_workers', default=8)
        parser.add_argument('--manual_seed', default=7)
        parser.add_argument('--size', default=128)

        parser.add_argument('--cm_prob', default=0.5, help='Cutmix probability')
        parser.add_argument('--cm_beta', default=1.0)
        parser.add_argumnet('--blur_prob', default=0.5, help='Gaussian probability')
        parser.add_argument('--blog_sig', default=0.5, help='Gaussian sigma')
        parser.add_argument('--jpg_prob', default=0.5, help='JPEG compression')
        parser.add_argument('--fc_name', default='_fc.')
        
        parser.add_argument('--gpu_id', default=0)
        
        parser.add_argument('--pretrained_dir', type=str, default='')
        parser.add_argument('--resume_dir', type=str, default='')
        self.initialized = True
        return parser