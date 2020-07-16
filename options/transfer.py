import argparse
import os

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_arugment('--target', required=True, help='pggan, star, style1, style2, bed, bedroom')
        parser.add_argument('--source_dataset', type=str, help='dataset dir')
        parser.add_argument('--target_dataset', type=str, help='dataset dir')
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='efficientnet-b0', help='architecture for binary classification')
        parser.add_argument('--checkpoint', default='./log/')

        # data augmentation
        parser.add_argument('--classes', default=2)
        parser.add_argument('--epochs', default=400)
        parser.add_argument('--iterations', default=500)
        parser.add_argument('--start_epoch', default=0)
        parser.add_argument('--train_batch', default=200)
        parser.add_argument('--test_batch', default=200)
        parser.add_argument('--lr', default=0.04)
        parser.add_argument('--schedule', default=[250, 250])
        parser.add_argument('--momentum', default=0.1)
        parser.add_argument('--gamma', default=0.1)
        parser.add_argumnet('--s', default=1.0)

        parser.add_argument('--num_workers', default=8)
        parser.add_argument('--manual_seed', default=7)
        parser.add_argument('--size', default=128)
        
        parser.add_argumnet('--dropout', default=0.2, help='Dropout probability')
        parser.add_argumnet('--dropconnect', default=0, help='Dropconnect probability')

        parser.add_argument('--cm_prob', default=0.5, help='Cutmix probability')
        parser.add_argument('--cm_beta', default=1.0)
        parser.add_argument('--blur_prob', default=0.5, help='Gaussian probability')
        parser.add_argument('--blog_sig', default=0.5, help='Gaussian sigma')
        parser.add_argument('--jpg_prob', default=0.5, help='JPEG compression')
        parser.add_argument('--fc_name', default='_fc.')
        
        parser.add_argument('--gpu_id', default=0)
        
        parser.add_argument('--pretrained_dir', type=str, default='')
        parser.add_argument('--resume_dir', type=str, default='')
        self.initialized = True
        return parser


    def gather_options(self):
            # initialize parser with basic options
            if not self.initialized:
                parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                parser = self.initialize(parser)

            # get the basic options
            opt, _ = parser.parse_known_args()
            self.parser = parser

            return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        if print_options:
            self.print_options(opt)

        self.opt = opt
        return self.opt