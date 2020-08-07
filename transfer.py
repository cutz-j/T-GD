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

from model_pytorch import EfficientNet
from utils import Bar,Logger, AverageMeter, accuracy, mkdir_p, savefig
from warmup_scheduler import GradualWarmupScheduler
from utils.aug import data_augment, rand_bbox
from utils.train_utils import save_checkpoint, adjust_learning_rate

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from options.transfer import BaseOptions

opt = BaseOptions().parse(print_options=False)
print("{} from {} model testing on {}".format(opt.arch, opt.source_dataset, opt.target_dataset))

gpu_id = opt.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
use_cuda = torch.cuda.is_available()
print("GPU device %d:" %(gpu_id), use_cuda)
best_acc = 0

if not os.path.isdir(opt.checkpoint):
    os.makedirs(opt.checkpoint)
    
train_dir = os.path.join(opt.source_dir, os.path.joint(opt.target, '2000_shot'))
val_target_dir = os.path.join(opt.target_dir, 'val')
val_source_dir = os.path.join(opt.source_dir, 'val')

train_aug = transforms.Compose([
    transforms.Lambda(lambda img: data_augment(img)),
    transforms.Resize(opt.size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_aug = transforms.Compose([
    transforms.Resize(opt.size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# pin_memory : cuda pin memeory use
train_loader = DataLoader(datasets.ImageFolder(train_dir, transform=train_aug),
                          batch_size=opt.train_batch, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
val_target_loader = DataLoader(datasets.ImageFolder(val_target_dir, val_aug),
                       batch_size=opt.test_batch, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
val_source_loader = DataLoader(datasets.ImageFolder(val_source_dir, val_aug),
                       batch_size=opt.test_batch, shuffle=True, num_workers=opt.num_workers, pin_memory=True)


teacher_model = EfficientNet.from_name(opt.arch, num_classes=opt.num_classes,
                              override_params={'dropout_rate':opt.dropout, 'drop_connect_rate':opt.dropconnect})
student_model = EfficientNet.from_name(opt.arch, num_classes=opt.num_classes,
                              override_params={'dropout_rate':opt.dropout, 'drop_connect_rate':opt.dropconnect})

# Pre-trained
if opt.pretrained:
    print("=> using pre-trained model '{}'".format(opt.pretrained))
    teacher_model.load_state_dict(torch.load(opt.pretrained)['state_dict'])
    student_model.load_state_dict(torch.load(opt.pretrained)['state_dict'])

teacher_model.to('cuda')
student_model.to('cuda')
cudnn.benchmark = True
print('    Total params: %.2fM' % (sum(p.numel() for p in teacher_model.parameters())/1000000.0))

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(student_model.parameters(), lr=opt.lr, momentum=opt.momentum)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)
# scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=2, total_epoch=50, after_scheduler=scheduler_cosine)
scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250)

# Resume
if opt.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = os.path.dirname(opt.resume)
    resume = torch.load(opt.resume)
    best_acc = resume['best_acc']
    start_epoch = resume['epoch']
    student_model.load_state_dict(resume['state_dict'])
    optimizer.load_state_dict(resume['optimizer'])
    logger = Logger(os.path.join(checkpoint, 'log.txt'), resume=True)
else:
    logger = Logger(os.path.join(checkpoint, 'log.txt'))
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Source Loss', 'Train AUROC', 'Valid AUROC', 'Source AUROC'])

# save teacher model weights
for param in teacher_model.parameters():
    param.requires_grad = False
teacher_model.eval()

teacher_model_weights = {}
for name, param in teacher_model.named_parameters():
    teacher_model_weights[name] = param.detach()

# L2-reg & L2-norm
def reg_cls(model):
    l2_cls = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if name.startswith(opt.fc_name):
            l2_cls += 0.5 * torch.norm(param) ** 2
    return l2_cls

def reg_l2sp(model):
    sp_loss = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if not name.startswith(opt.fc_name):
            sp_loss += 0.5 * torch.norm(param - teacher_model_weights[name]) ** 2
    return sp_loss

def train(opt, train_loader, teacher_model, student_model, criterion, optimizer, epoch, use_cuda):
    teacher_model.eval()
    student_model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    arc = AverageMeter()
    end = time.time()
    cls_losses = AverageMeter()
    sp_losses = AverageMeter()
    main_losses = AverageMeter()
    alpha = AverageMeter()
    
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
        teacher_outputs = teacher_model(inputs)
        teacher_loss = criterion(teacher_outputs, targets)
        sp_gamma = 0
        sigmoid = nn.Sigmoid()
        sp_gamma += opt.s*sigmoid(-teacher_loss)
        
        outputs = student_model(inputs)
        loss_main = criterion(outputs, targets)
        loss_cls = 0
        loss_sp = 0
        loss_cls = reg_cls(student_model)
        loss_sp = reg_l2sp(student_model)
        loss =  loss_main + sp_gamma*loss_sp + sp_gamma*loss_cls

        # measure accuracy and record loss
        losses.update(loss.data.tolist(), inputs.size(0))
        cls_losses.update(loss_cls, inputs.size(0))
        sp_losses.update(loss_sp, inputs.size(0))
        main_losses.update(loss_main.tolist(), inputs.size(0))
        alpha.update(sp_gamma, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    print('Train | {batch}/{size} | Loss:{loss:.4f} | MainLoss:{main:.4f} | Alpha:{alp:.4f} | SPLoss:{sp:.4f} | CLSLoss:{cls:.4f} | AUROC:{ac:.4f}'.format(
                     batch=batch_idx+1, size=len(train_loader), loss=losses.avg, main=main_losses.avg, alp=alpha.avg, sp=sp_losses.avg, cls=cls_losses.avg, ac=arc.avg))
    return (losses.avg, arc.avg)

def test(opt, val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    arc = AverageMeter()
    cls_losses = AverageMeter()
    sp_losses = AverageMeter()
    main_losses = AverageMeter()


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
            loss_main = criterion(outputs, targets)
            loss_cls = 0
            loss_sp = 0
            loss_cls = reg_cls(model)
            loss_sp = reg_l2sp(model)
            loss = loss_main + 0*loss_sp + 0*loss_cls

            # measure accuracy and record loss
            auroc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy()[:,1])
            losses.update(loss.data.tolist(), inputs.size(0))
            arc.update(auroc, inputs.size(0))
            cls_losses.update(loss_cls, inputs.size(0))
            sp_losses.update(loss_sp, inputs.size(0))
            main_losses.update(loss_main.tolist(), inputs.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('Test | {batch}/{size} | Loss:{loss:.4f} | MainLoss:{main:.4f} | SPLoss:{sp:.4f} | CLSLoss:{cls:.4f} | AUROC:{ac:.4f}'.format(
                     batch=batch_idx+1, size=len(train_loader), loss=losses.avg, main=main_losses.avg, sp=sp_losses.avg, cls=cls_losses.avg, ac=arc.avg))
    return (losses.avg, arc.avg)

for epoch in range(start_epoch, opt.epochs):
    opt.lr = optimizer.state_dict()['param_groups'][0]['lr']

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.lr))
    
    train_loss, train_auroc = train(opt, train_loader, teacher_model, student_model, criterion, optimizer, epoch, use_cuda)
    test_loss, test_auroc = test(opt, val_target_loader, student_model, criterion, epoch, use_cuda)
    source_loss, source_auroc = test(opt, val_source_loader, student_model, criterion, epoch, use_cuda)


    logger.append([opt.lr, train_loss, test_loss,  source_loss, train_auroc, test_auroc, source_auroc])
    is_best = test_auroc+source_auroc > best_acc
    best_acc = max(test_auroc+source_auroc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict' : student_model.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint=checkpoint)
    scheduler_cosine.step()
    scheduler_step.step()
    
    if (epoch+1)%200 == 0:
        teacher_model.load_state_dict(student_model.state_dict())