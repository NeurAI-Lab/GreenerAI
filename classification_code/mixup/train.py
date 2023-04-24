# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
from datetime import datetime
from modules.mixup import mixup_data, mixup_criterion

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
from utils import get_network, get_optimizer, get_training_dataloader, get_test_dataloader, WarmUpLR, count_parameters
from energy_meter import EnergyMeter
from contextlib import ExitStack
from random import uniform

total = 0
train_total = 0
val_total = 0


def train(epoch):
    net.train()

    global train_total
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    loop = tqdm(cifar100_training_loader)
    for batch_index, (images, labels) in enumerate(loop):
        if args.scheduler == "half-poly":
            train_scheduler.step()
        elif epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        inputs=images
        if (args.mixup_alpha is not None):
            inputs, targets_a, targets_b, lam = mixup_data(images, labels, args.mixup_alpha)
        start.record()
        outputs = net(inputs)

        end.record()
        torch.cuda.synchronize()
        time_elapsed = start.elapsed_time(end)
        train_total += time_elapsed
        if (args.mixup_alpha is not None):
            loss = mixup_criterion(loss_function, outputs, targets_a, targets_b, lam)
        else:
            loss = loss_function(outputs, labels)
        if args.random_gradient:
            loss *= uniform(0, 1)
        loss.backward()

        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-checkname', type=str, required=True, help='experiment name')
    parser.add_argument('-optimizer', type=str, default="sgd", help='choose optimizer')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-output_dir', default='./runs', help='path where to save')
    parser.add_argument('-loss', default='cross_entropy', help='path where to save')
    parser.add_argument('-random_gradient', action="store_true", help='apply random gradient')
    parser.add_argument('-em', default=True, help='METER ENERGY')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument( "-mixup_alpha", type=float,default=None,help="Alpha value; None = no mixup",)
    parser.add_argument('-scheduler', default='stepwise', help='scheduler type')

    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)

    # data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=0,
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

    if args.loss == "cross_entropy":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise ValueError

    optimizer = get_optimizer(args.optimizer, params=net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.scheduler=="stepwise":
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)
    elif args.scheduler=="half-poly":
        settings.EPOCH=int(settings.EPOCH/2 )
        train_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (len(cifar100_training_loader) * settings.EPOCH)) ** 0.9)
    # learning rate decay
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

    with EnergyMeter(writer=writer, dir=log_path) if args.em else ExitStack():
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        best_acc = 0.0
        best_epoch = -1
        for epoch in range(1, settings.EPOCH):
            if epoch > args.warm and  args.scheduler !="half-poly":
                train_scheduler.step(epoch)

            train(epoch)
            acc = eval_training(epoch)

            # start to save best performance model after learning rate decay to 0.01
            if best_acc < acc:
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
    print("total train+val(Gpu only)", str(train_total + val_total))
    print("total train+val(cpu+Gpu)", str(total_time))
    print("#########################################")
    print("Best Acc: ", best_acc)
    print("Best Epoch: ", best_epoch)
