import torch
import torch.nn as nn
import torch.nn.functional as nfunc

import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True)


def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(nfunc.relu(self.bn1(x))))
        out = self.conv2(nfunc.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, config):
        super(WideResNet, self).__init__()
        depth, widen_factor, dropout_rate, num_classes, size = config
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        n_stages = [16, int(16 * k), int(32 * k), int(64 * k), int(128 * k)]

        self.conv1 = conv3x3(3, n_stages[0])
        self.layer1 = self._wide_layer(WideBasic, n_stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, n_stages[2], n, dropout_rate, stride=2)

        self.layer3 = self._wide_layer(WideBasic, n_stages[3], n, dropout_rate, stride=2)
        if size >= 64:
            self.layer4 = self._wide_layer(WideBasic, n_stages[4], n, dropout_rate, stride=2)
        i = 4 if size >= 64 else 3
        self.bn1 = nn.BatchNorm2d(n_stages[i], momentum=0.9)
        self.linear = nn.Linear(n_stages[i], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        if x.shape[-1] >= 32:
            out = self.layer3(out)
            if x.shape[-1] == 64:
                out = self.layer4(out)

        out = nfunc.relu(self.bn1(out))
        out = nfunc.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


if __name__ == '__main__':
    net = WideResNet(28, 10, 0.3, 10)
    y = net(torch.randn(1, 3, 32, 32))

    print(y.size())
