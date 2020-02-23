import os
import sys
import torch
from torch import nn
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
from efficientnet import EfficientNet
from resnext import resnext50_32x4d
from options.test import TestOptions
from utils import Bar,Logger, AverageMeter, accuracy, mkdir_p, savefig
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



opt = TestOptions().parse(print_options=False)
print("{} from {} model testing on {}".format(opt.arch, opt.source_dataset, opt.target_dataset))

gpu_id = opt.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
use_cuda = torch.cuda.is_available()
print("GPU device %d:" %(gpu_id), use_cuda)

model = EfficientNet.from_name(opt.arch, num_classes=opt.classes)

if opt.resume:
    pretrained = opt.resume
    print("=> using pre-trained model '{}'".format(pretrained))
    model.load_state_dict(torch.load(pretrained)['state_dict'])
    
model.to('cuda')
cudnn.benchmark = True
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

def test(val_loader, model, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    arc = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader)):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets.data)
            losses.update(loss.data.tolist(), inputs.size(0))
            auroc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy()[:,1])
            arc.update(auroc, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('{batch}/{size} | Loss:{loss:} | AUROC:{ac:}'.format(
         batch=batch_idx+1, size=len(val_loader), loss=losses.avg, ac=arc.avg))
    return (losses.avg, arc.avg)

test_aug = transforms.Compose([
    transforms.Resize(opt.size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    

data_dir = opt.source_dataset
test_dir = os.path.join(data_dir, 'test')
test_loader = DataLoader(datasets.ImageFolder(test_dir, test_aug),
                       batch_size=opt.test_batch, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

print("Performance of {}".format(data_dir))
test_loss, test_auroc = test(test_loader, model, criterion, 1, use_cuda)

data_dir = opt.target_dataset
test_dir = os.path.join(data_dir, 'test')
test_loader = DataLoader(datasets.ImageFolder(test_dir, test_aug),
                       batch_size=opt.test_batch, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

print("Performance of {}".format(data_dir))
test_loss, test_auroc = test(test_loader, model, criterion, 1, use_cuda)
