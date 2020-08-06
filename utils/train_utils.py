import os
import torch
import shutil


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, opt):
    lr_set = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    lr_list = opt.schedule.copy()
    lr_list.append(epoch)
    lr_list.sort()
    idx = lr_list.index(epoch)
    opt['lr'] *= lr_set[idx]
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt['lr']