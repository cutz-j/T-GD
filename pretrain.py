import os
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from models import EfficientNet
from utils import Bar,Logger, AverageMeter, accuracy, mkdir_p, savefig
from warmup_scheduler import GradualWarmupScheduler
from utils.aug import data_augment, rand_bbox
from utils.train_utils import save_checkpoint, adjust_learning_rate

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from options.base import BaseOptions

opt = BaseOptions().parse(print_options=False)
#print("{} from {} model testing on {}".format(opt.arch, opt.source_dataset, opt.target_dataset))

gpu_id = opt.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
use_cuda = torch.cuda.is_available()
print("GPU device %d:" %(gpu_id), use_cuda)

model = EfficientNet.from_name(opt.arch, num_classes=opt.classes,
                              override_params={'dropout_rate': opt.dropout, 'drop_connect_rate': opt.dropconnect})
    
model.to('cuda')
cudnn.benchmark = True

data_dir = opt.source_dataset
train_dir = os.path.join(data_dir, 'train')
train_aug = transforms.Compose([
    transforms.Lambda(lambda img: data_augment(img, opt)),
    transforms.Resize(opt.size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
#     transforms.RandomErasing(p=0.3, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0, inplace=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_loader = DataLoader(datasets.ImageFolder(train_dir, train_aug),
                       batch_size=opt.train_batch, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

val_dir = os.path.join(data_dir, 'val')
val_aug = transforms.Compose([
    transforms.Resize(opt.size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_loader = DataLoader(datasets.ImageFolder(val_dir, val_aug),
                       batch_size=opt.test_batch, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=1e-4)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler_cosine)

# Resume
if opt.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = os.path.dirname(opt.resume)
#     checkpoint = torch.load(resume)
    resume = torch.load(opt.resume)
    best_acc = resume['best_acc']
    start_epoch = resume['epoch']
    model.load_state_dict(resume['state_dict'])
    optimizer.load_state_dict(resume['optimizer'])
    logger = Logger(os.path.join(checkpoint, 'log.txt'), resume=True)
else:
    logger = Logger(os.path.join(opt.checkpoint, 'log.txt'))
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Train AUROC.', 'Valid AUROC.'])

def train(opt, train_loader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    arc = AverageMeter()
    end = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_size = inputs.size(0)
        if batch_size < opt.train_batch:
            continue
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            
        r = np.random.rand(1)
        if opt.cm_beta > 0 and r < opt.cm_prob:
            
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            tt= targets[rand_index]
            boolean = targets==tt
            rand_index = rand_index[boolean]
            lam = np.random.beta(opt.cm_beta, opt.cm_beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[boolean, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data)
        auroc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy()[:,1])
        losses.update(loss.data.tolist(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        arc.update(auroc, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % 100 == 0:
            print('{batch}/{size} | Loss:{loss:.4f} | top1:{tp1:.4f} | AUROC:{ac:.4f}'.format(
                     batch=batch_idx+1, size=len(train_loader), loss=losses.avg, tp1=top1.avg, ac=arc.avg))
    print('{batch}/{size} | Loss:{loss:.4f} | top1:{tp1:.4f} | AUROC:{ac:.4f}'.format(
                     batch=batch_idx+1, size=len(train_loader), loss=losses.avg, tp1=top1.avg, ac=arc.avg))
    return (losses.avg, top1.avg, arc.avg)

def test(opt, val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    arc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets.data)
            auroc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy()[:,1])
            losses.update(loss.data.tolist(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            arc.update(auroc, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('{batch}/{size} | Loss:{loss:.4f} | top1:{tp1:.4f} | AUROC:{ac:.4f}'.format(
         batch=batch_idx+1, size=len(val_loader), loss=losses.avg, tp1=top1.avg, ac=arc.avg))
    return (losses.avg, top1.avg, arc.avg)


for epoch in range(opt.start_epoch, opt.epochs):
    opt.lr = optimizer.state_dict()['param_groups'][0]['lr']
    adjust_learning_rate(optimizer, epoch)
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.lr))
    
    train_loss, train_acc, train_auroc = train(opt, train_loader, model, criterion, optimizer, epoch, use_cuda)
    test_loss, test_acc, test_auroc = test(opt, val_loader, model, criterion, epoch, use_cuda)
    
    logger.append([opt.lr, train_loss, test_loss, train_acc, test_acc, train_auroc, test_auroc])
    scheduler_warmup.step()

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict' : model.state_dict(),
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint=opt.checkpoint)
