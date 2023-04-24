# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import copy
import os
import sys
import argparse
from datetime import datetime
from loss.focal_loss import FocalLoss
from loss.class_balanced_loss import ClassBalancedLosses
from lr_scheduler import LR_Scheduler as LR_Scheduler_V2
from lr_finder_v2 import LRFinder as LRFinder_V2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
# from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from tqdm import tqdm
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, count_parameters
from energy_meter import EnergyMeter
from contextlib import ExitStack
from random import uniform
from modules.radam import RAdam
from modules.ranger import Ranger

total = 0
train_total=0
val_total=0
def get_optimizer(name, **kwargs):
    if name == "sgd":
        optimizer = optim.SGD(**kwargs)
    elif name == "radam":
        kwargs.pop('momentum', None)
        optimizer = RAdam(**kwargs)
    elif name == "ranger":
        kwargs.pop('momentum', None)
        optimizer = Ranger(**kwargs)
    else:
        raise ValueError("Unkown Optimizer")
    return optimizer

def train(epoch):
    net.train()

    global train_total
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    loop = tqdm(cifar100_training_loader)
    prev_state = copy.deepcopy(net.state_dict())
    for batch_index, (images, labels) in enumerate(loop):
        if epoch <= args.warm:
            warmup_scheduler.step()
        if args.lr_scheduler =='clr':
            lr=train_scheduler(optimizer, batch_index, epoch)
            writer.add_scalar(
                "xtras/learning_rate", lr, epoch * len(cifar100_training_loader) + batch_index + 1
            )
        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()

        start.record()

        outputs = net(images)

        end.record()
        torch.cuda.synchronize()
        time_elapsed = start.elapsed_time(end)
        train_total += time_elapsed

        loss = loss_function(outputs, labels)
        if args.random_gradient:
            loss *= uniform(0, 1)
        loss.backward()

        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
        if args.lookAhead_steps != 0 and batch_index % args.lookAhead_steps == args.lookAhead_steps - 1:
            curr_state = net.state_dict()
            for key in curr_state.keys():
                curr_state[key] += (1.0 / args.lookAhead_alpha - 1) * prev_state[key]
                curr_state[key] *= args.lookAhead_alpha
            prev_state = copy.deepcopy(curr_state)

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        loop.set_description(
            'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


def eval_training(epoch):
    net.eval()
    global val_total
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        start.record()

        outputs = net(images)

        end.record()
        torch.cuda.synchronize()
        time_elapsed = start.elapsed_time(end)
        val_total += time_elapsed

        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    print()

    # add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

def findlr(losses,optimizer,trainloader,model):
    loss=dict()
    loss["loss"]=losses
    criterion = loss
    lr_finder = LRFinder_V2(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(
        trainloader, end_lr=10, num_iter=5000, step_mode="exp"
    )
    lr_finder.plot(log_lr=True)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-loss', type=str, default="cross_entropy", help='loss type:options[focal,cross_entropy] default :cross_entropy')
    parser.add_argument('-optimizer', type=str, default="sgd", help='optimizer type:options[sgd,radam] default :sgd')
    parser.add_argument('-checkname', type=str, required=True, help='experiment name')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-output_dir', default='./runs', help='path where to save')
    parser.add_argument('-random_gradient', action="store_true", help='apply random gradient')
    parser.add_argument('-em', default=True, help='METER ENERGY')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-clr_max', type=float, default=0.01, help='max clr ')
    parser.add_argument('-lr_scheduler', type=str, default='default', help=' learning rate scheduler')
    parser.add_argument('-lookAhead_steps', type=int, default=0, help='lookAhead steps')
    parser.add_argument('-lookAhead_alpha', type=float, default=.5, help='lookAhead alpha')

    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)

    # data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=8,
        batch_size=args.b,
        shuffle=args.s
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=0,
        batch_size=args.b,
        shuffle=args.s
    )

    if args.loss=="focal":
        loss_function=FocalLoss()
    elif args.loss=="cross_entropy":
        loss_function = nn.CrossEntropyLoss()
    elif args.loss == "class_balanced":
        loss_function = ClassBalancedLosses(100)
    else:
        assert False,"not implemented loss function"
    optimizer = get_optimizer(args.optimizer, params=net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.lr_scheduler == "clr":
        train_scheduler = LR_Scheduler_V2(
            args.lr_scheduler,
            args.lr,
            settings.EPOCH,
            len(cifar100_training_loader),
            stepsize=len(cifar100_training_loader) * 20,
            max_lr=args.clr_max,
        )
    else:
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    if args.warm:
        iter_per_epoch = len(cifar100_training_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    log_path = os.path.join(args.output_dir, args.net, args.checkname)

    writer = SummaryWriter(log_dir=os.path.join(log_path, "summary"))
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    checkpoint_path = os.path.join(log_path, '{net}-{epoch}-{type}.pth')
    print("################################################")
    print("## Number of Parameters                        #")
    print("##", count_parameters(model=net), "                                   #")
    print("################################################")

    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    # findlr(loss_function, optimizer, cifar100_training_loader, net)
    # iiiii=0

    with EnergyMeter(writer=writer, dir=log_path) if args.em else ExitStack():
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        best_acc = 0.0
        best_epoch = -1
        for epoch in range(1, settings.EPOCH):
            if epoch > args.warm and args.lr_scheduler != "clr":
                train_scheduler.step(epoch)

            train(epoch)
            acc = eval_training(epoch)

            # start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                best_acc = acc
                best_epoch = epoch
                continue

            if not epoch % settings.SAVE_EPOCH:
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

        writer.close()

    total_end.record()
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)

    print("total train(Gpu only)", str(train_total))
    print("total val(Gpu only)", str(val_total))
    print("total train+val(Gpu only)", str(train_total+val_total))
    print("total train+val(cpu+Gpu)", str(total_time))
    print("#########################################")
    print("Best Acc: ", best_acc)
    print("Best Epoch: ", best_epoch)
