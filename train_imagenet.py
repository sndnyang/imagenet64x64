import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data.sampler as sampler
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm, trange
import numpy as np
import tensorboardX as tbX

from dllib.ExpUtils import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', metavar='DIR', default='image64', help='dataset')
parser.add_argument('--data-dir', metavar='DIR', default='data', help='path to dataset')
parser.add_argument('--size', default=128000, type=int, metavar='N', help='number of data, 0->all (default: 128000)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--max-iter', default=40000, type=int)
parser.add_argument('--lr-drop-iter', nargs="+", default=[40000//3, 40000*2//3, 40000*8//9])
parser.add_argument('--eval-iter', default=500, type=int)
parser.add_argument('--print-freq', default=1000, type=int)
parser.add_argument('--warmup', action='store_true')
parser.add_argument('--warmup-iter', type=int, default=40000*5//90)
parser.add_argument('-bu', '--batch-size-unlabeled', default=0, type=int)
parser.add_argument('-ui', '--unlabeled-iter', default=30, type=int)
parser.add_argument('-b',  '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--log-dir', type=str, default='', metavar='S', help='tensorboard directory, (default: an absolute path)')
parser.add_argument('--log-arg', type=str, default='', metavar='S', help='show the arguments in directory name')
parser.add_argument('--debug', action='store_true', default=False, help='compare log side by side')
parser.add_argument('--vis', action='store_true', default=False, help='visual by tensor board')

best_acc1 = 0


def relabel_dataset(dataset, n_l=13000):
    labels = np.array(dataset.targets)
    if n_l == 0:
        return np.arange(labels.shape[0])
    labeled_idx = []
    s_c = int(n_l / 1000)
    for i in range(1000):
        ind = np.where(labels == i)[0]
        labeled_idx.append(np.random.choice(ind, s_c))
    labeled_idx = np.concatenate(labeled_idx)
    return labeled_idx


def main():
    args = parser.parse_args()
    args.lr_drop_iter = [int(val) for val in args.lr_drop_iter]
    if not args.log_arg:
        args.log_arg = "ce-arch-lr-size-batch_size-uda"
    if args.vis:
        args.dir_path = form_dir_path("L0VAT-semi", args)
        set_file_logger(logger, args)
        args.writer = tbX.SummaryWriter(log_dir=args.dir_path)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data_dir, args.dataset, 'train')
    valdir = os.path.join(args.data_dir, args.dataset, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # train_labeled_dataset = ImageNet(
    #     traindir, args,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]),
    #     db_path='./data_split/labeled_images_0.10.pth',
    #     )
    #
    # train_unlabeled_dataset = ImageNet(
    #     traindir, args,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]),
    #     db_path='./data_split/unlabeled_images_0.90.pth',
    #     is_unlabeled=True,
    #     )

    normalize = transforms.Normalize(mean=[0.482, 0.458, 0.408],
                                     std=[0.269, 0.261, 0.276])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    labeled_idx = relabel_dataset(train_dataset, args.size)
    print("N_train_labeled:{}".format(labeled_idx.size))
    print("mean %f, max %f" % (labeled_idx.mean(), labeled_idx.max()))
    l_sampler = sampler.SubsetRandomSampler(labeled_idx)
    train_sampler = sampler.BatchSampler(l_sampler, args.batch_size, drop_last=False)

    train_labeled_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.workers, pin_memory=True)
    train_unlabeled_loader = None

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    entropy_criterion = HLoss()

    iter_sup = iter(train_labeled_loader)
    if train_unlabeled_loader is None:
        iter_unsup = None
    else:
        iter_unsup = iter(train_unlabeled_loader)

    model.train()
    meters = initialize_meters()
    for train_iter in trange(args.max_iter):

        lr = adjust_learning_rate(optimizer, train_iter + 1, args)
        try:
            batch_data = next(iter_sup)
        except StopIteration:
            iter_sup = iter(train_labeled_loader)
            batch_data = next(iter_sup)

        un_batch_data = None
        if iter_unsup is not None:
            try:
                un_batch_data = next(iter_unsup)
            except StopIteration:
                iter_unsup = iter(train_unlabeled_loader)
                un_batch_data = next(iter_unsup)

        train(batch_data, model, optimizer, criterion, un_batch_data, entropy_criterion, meters, args)

        if (train_iter+1) % args.print_freq == 0:
            mes = 'ITER: [{0}/{1}]  Data time {data_time.val:.3f} ({data_time.avg:.3f}) Time {batch_time.val:.3f} ({batch_time.avg:.3f}) ' \
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  HLoss {h_loss.val:.4f} ({h_loss.avg:.4f})  Unsup Loss {unsup_loss.val:.4f} ({unsup_loss.avg:.4f})' \
                  '  Prec@1 {top1.val:.3f} ({top1.avg:.3f})   Prec@5 {top5.val:.3f} ({top5.avg:.3f})   Learning rate {2}' \
                .format(train_iter+1, args.max_iter, lr, data_time=meters['data_time'], batch_time=meters['batch_time'],
                        loss=meters['losses'], h_loss=meters['losses_entropy'], unsup_loss=meters['losses_unsup'],
                        top1=meters['top1'],  top5=meters['top5'])
            tqdm.write(mes)
        if (train_iter+1) % args.eval_iter == 0:
            # evaluate on validation set
            acc1, acc5 = validate(val_loader, model, criterion, args)
            if args.vis:
                dicts = {"Test/Acc": acc1, "Test/Top5Acc": acc5}
                vis_step(args.writer, (train_iter + 1) / args.eval_iter, dicts)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
         
            save_checkpoint({
                'iter': train_iter + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.dir_path)
            model.train()
            meters = initialize_meters()


def initialize_meters():
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_entropy = AverageMeter('Loss entropy', ':.4e')
    losses_unsup = AverageMeter('Loss unsup', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    return {'batch_time':batch_time,
            'data_time':data_time,
            'losses':losses,
            'losses_entropy':losses_entropy,
            'losses_unsup':losses_unsup,
            'top1':top1,
            'top5':top5}


def train(iter_sup, model, optimizer, criterion, iter_unsup, entropy_criterion, meters, args):

    t0 = time.time()
    images, target = iter_sup

    images, target = images.cuda(), target.cuda()
    data_time = time.time()-t0

    output = model(images)

    loss_all = 0
    loss_cls = criterion(output, target)
    meters['losses'].update(loss_cls.item(), images.size(0))
    loss_cls.backward()
    loss_all += loss_cls.item()

    if iter_unsup is not None:
        '''
            LOSSES for unlabeled samples
        '''
        #TODO 
        #images_unlabeled_all, images_unlabeled_all_aug = next(iter_unsup)
        #images_unlabeled_all, images_unlabeled_all_aug = images_unlabeled_all.cuda(), images_unlabeled_all_aug.cuda()
        #sub_batch_count = -( - images_unlabeled_all.size(0) // args.batch_size )
        loss_unsup_all = 0
        for sub_batch_idx in range(args.unlabeled_iter):
            t1 = time.time()
            images_unlabeled, images_unlabeled_aug = iter_unsup
            data_time += time.time() - t1
            images_unlabeled, images_unlabeled_aug = images_unlabeled.cuda(), images_unlabeled_aug.cuda()
            #images_unlabeled = images_unlabeled_all[ sub_batch_idx*args.batch_size: min( (sub_batch_idx+1)*args.batch_size, images_unlabeled_all.size(0))]
            #images_unlabeled_aug = images_unlabeled_all_aug[ sub_batch_idx*args.batch_size: min( (sub_batch_idx+1)*args.batch_size, images_unlabeled_all.size(0))]
            with torch.no_grad():
                output_unlabeled = model(images_unlabeled)
            output_unlabeled_aug = model(images_unlabeled_aug)
 
            # Technique 1: entropy loss for augmented images
            entropy_weight = 1.0
            loss_entropy = entropy_weight * entropy_criterion(output_unlabeled_aug)
            meters['losses_entropy'].update(loss_entropy.item(), images_unlabeled.size(0))
  
            # Technique 2: Softmax temperature control for unsupervised loss
            temperature = 0.4
            loss_kl = torch.nn.functional.kl_div( 
                                    torch.nn.functional.log_softmax(output_unlabeled_aug, dim=1), 
                                    torch.nn.functional.softmax(output_unlabeled / temperature, dim=1).detach(),
                                    reduction='none')
  
            # Technique 3: confidence-based masking
            threshold = 0.5
            max_y_unlabeled = torch.max( torch.nn.functional.softmax( output_unlabeled / temperature, dim=1 ), 1, keepdim=True )[0]
            mask = (max_y_unlabeled > threshold).type(torch.cuda.FloatTensor)
  
            loss_kl = torch.sum(loss_kl * mask) / (mask.mean()+1e-8)
  
            loss_unsup = loss_entropy + loss_kl
            unsup_loss_weight = 20.0
            loss_unsup = loss_unsup / args.unlabeled_iter * unsup_loss_weight
            loss_unsup.backward()
            loss_unsup_all += loss_unsup.item()
        meters['losses_unsup'].update( loss_unsup_all, args.batch_size_unlabeled * args.unlabeled_iter )

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    meters['top1'].update(acc1[0], images.size(0))
    meters['top5'].update(acc5[0], images.size(0))

    # compute gradient and do SGD step
    # loss_all.backward()
    optimizer.step()
    optimizer.zero_grad()

    # measure elapsed time
    meters['batch_time'].update(time.time() - t0 - data_time)
    meters['data_time'].update(data_time)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    tqdm.write('Val Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    wlog('Val Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg, top5.avg


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, train_iter, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if train_iter <= args.warmup_iter and args.warmup:
        # warmup
        lr = args.lr * ( float(train_iter) / float(args.warmup_iter) )
    elif train_iter < args.lr_drop_iter[0]:
        lr = args.lr
    elif train_iter >= args.lr_drop_iter[0] and train_iter < args.lr_drop_iter[1]:
        lr = args.lr * 0.1
    elif train_iter >= args.lr_drop_iter[1] and train_iter < args.lr_drop_iter[2]:
        lr = args.lr * 0.01
    elif train_iter >= args.lr_drop_iter[2]:
        lr = args.lr * 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
