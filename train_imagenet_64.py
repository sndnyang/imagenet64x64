import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
from six.moves import cPickle as pickle

from dllib.ExpUtils import get_hms
from dllib.torch_func.evaluate import AverageMeter, accuracy
import torchvision.models as models

from main import adjust_learning_rate

parser = argparse.ArgumentParser(description='PyTorch Wide-ResNet Training')
parser.add_argument('--gpu-ids', default='0', type=str, help='gpu id list')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--optimizer', default='SGD', type=str, help='SGD/Adam/RMSprop')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--arch', default='wrn', type=str, help='model')
parser.add_argument('--num-epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--batch-size', default=10, type=int, help='a batch of data into the model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--k', default=0.5, type=float, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

device = torch.device('cuda')
# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
num_epochs, batch_size, optim_type = args.num_epochs, args.batch_size, args.optimizer


# data_loader
def unpickle(file):
    with open(file, 'rb') as fo:
        data_instance = pickle.load(fo)
    return data_instance


def iterate_mini_batches(inputs, targets, batchsize, shuffle=False, augment=False, img_size=32):

    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping
            padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=8, size=(batchsize, 2))
            for r in range(batchsize):
                random_cropped[r, :, :, :] = \
                    padded[r, :, crops[r, 0]:(crops[r, 0]+img_size), crops[r, 1]:(crops[r, 1]+img_size)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]


def load_data_batch(data_folder, idx, img_size=64):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x / np.float32(255)
    mean_image = mean_image / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return dict(
        X_train=X_train,
        Y_train=Y_train.astype('int64'),
        mean=mean_image)


def load_validation_data(data_folder, mean_image, img_size=32):
    test_file = os.path.join(data_folder, 'val_data')

    d = unpickle(test_file)
    x = d['data']
    y = d['labels']
    x = x / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = np.array([i-1 for i in y])

    # Remove mean (computed from training data) from images
    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    return dict(
        X_test=x,
        Y_test=y.astype('int64'))


data_folder = '/home/xyang2/project/data/dataset/imagenet64'
num_classes = 1000


# Return network & file name
def init_model(args):
    arch = models.__dict__[args.arch]()
    # arch = WideResNet(args.depth, args.k, args.dropout, num_classes, size=64)
    file_name = 'resnet'
    return arch, file_name


print('| Building net type [' + args.arch + ']...')
model, saved_file = init_model(args)
# model.apply(conv_init)


model.cuda()
c = range(torch.cuda.device_count())
model = torch.nn.DataParallel(model, device_ids=c)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()


# Training
def train(epoch, model, optimizer, scheduler):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    batch_idx = 0
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, scheduler.get_lr()[-1]))
    start = time.time()
    for batch in iterate_mini_batches(train_x, train_y, 16, shuffle=True, augment=True, img_size=img_size):
        batch_idx += 1
        inputs, targets = batch
        inputs, targets = torch.tensor(inputs), torch.tensor(targets)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum().item()

        end = time.time()
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%% Time: %.3fs' % (epoch, num_epochs, batch_idx+1, batch_idx+1, loss, 100.*correct/total, end-start))
        sys.stdout.flush()

    scheduler.step()


def test(epoch, model, args):
    global best_acc
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    end = time.time()
    for batch in iterate_mini_batches(test_x, test_y, 500, shuffle=False, img_size=img_size):
        inputs, targets = batch
        inputs, targets = torch.tensor(inputs), torch.tensor(targets)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, top_k=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)

    # Save checkpoint when best model
    acc = top1.avg
    # top1.avg, top5.avg, losses.avg
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%% Acc@5: %.2f%%" % (epoch, losses.avg, acc, top5.avg))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % acc)
        state = {
            'net': model.module if use_cuda else model,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/' + args.dataset + os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point + saved_file + '.t7')
        best_acc = acc


print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80], gamma=0.1)
train_x, train_y = [], []

img_size = 64
for i in range(1, 11):
    start_time_tmp = time.time()
    data = load_data_batch(data_folder, i, img_size=img_size)
    if i == 1:
        mean_image = data['mean']
    print('Data loading took %f' % (time.time() - start_time_tmp))
    train_x.append(data['X_train'])
    train_y.append(data['Y_train'])
    break

# Load test data
test_data = load_validation_data(data_folder, mean_image=mean_image, img_size=img_size)
test_x = test_data['X_test']
test_y = test_data['Y_test']
num_classes = 1000

train_x = np.concatenate(train_x)
train_y = np.concatenate(train_y)

for epoch in range(num_epochs):
    start_time = time.time()

    train(epoch, model, optimizer, scheduler)
    test(epoch, model, args)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % get_hms(elapsed_time))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' % best_acc)
