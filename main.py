'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from models import *
from utils import progress_bar
from utils import *
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', '-m', action = 'store_true',
                    help = 'test mode')

args = parser.parse_args()

cudnn.benchmark = True

print('parsed options:', vars(args))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
torch.randn(8).cuda()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
#epoch_step = json.loads(opt.epoch_step)


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
if args.mode:
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
else:
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = squeezenet.squeezenet1_0()
    net = alexnet()
    # net = squeezemob.squeezenet1_0()
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=args.lr) #, weight_decay=5e-4)
optimizer = optim.RMSprop(net.parameters(), lr=args.lr)

if args.mode:
    with open("tempos.txt", mode = 'w') as filetime:
        filetime.close()
else:
    with open("Loss.txt", mode = 'w') as fileloss:
        fileloss.close()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    seconds = seconds - millis/1000
    micros = int(seconds*1000000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if micros > 0 and i <= 2:
        f += str(micros) + 'us'
        i += 1
    if f == '':
        f = '0ms'
    return f

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/(batch_idx+1)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        initial_time = time.time()
        outputs = net(inputs)
        final_time = time.time()
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        time_passed = final_time - initial_time

        if args.mode:
            with open("tempos.txt", mode = 'a') as filetime:
                filetime.write('tempo: %s\n' % format_time(time_passed))
                filetime.close()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


if ~args.resume:
    print(torch_summarize(net))


loss = 0
delta = 0
drop = 0

if args.mode:
    test(1)
else:
    for epoch in range(200 - start_epoch):
        oldloss = loss
        loss = train(epoch)

        print('Epoch Loss: %s' % loss)
        with open("Loss.txt", mode = 'a') as fileloss:
            if epoch == 0:
                fileloss.write('EPOCH,LOSS,LR')
            fileloss.write('\n%d,%f,%f' % (epoch,loss,optimizer.param_groups[0]['lr']))

        test(epoch)
        lr = optimizer.param_groups[0]['lr']
        if (oldloss-loss < 0.01)and(epoch!=0):
            delta = delta+1
            if delta==10:
                optimizer.param_groups[0]['lr'] = lr*0.1
                drop = drop + 1
                delta = 0
        else: delta = 0
        if drop == 4:
            print('The end')
            print(lr, delta, drop, epoch)
            fileloss.close()
            break

