###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################


import os
import copy
import numpy as np
from tqdm import tqdm
from torch.nn import Parameter
import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import sys
from contextlib import ExitStack

sys.path.insert(0, "../../")

import encoding.utils as utils
from encoding_custom.nn.loss import (
    SegmentationLosses,
    ClassBalancedSegmentationLosses,
    ClassBalancedSegmentationLossesWithLabelRelaxation,
    OHEMSegmentationLosses,
)
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding_custom.datasets import get_segmentation_dataset
from encoding_custom.models import get_segmentation_model
from encoding_custom.nn.soft_dice_loss import SoftDiceLoss
from encoding_custom.nn.focal_loss import FocalLoss
from encoding_custom.utils.energy_meter import EnergyMeter
from experiments.segmentation.option import Options
from tensorboardX import SummaryWriter
from encoding.nn.syncbn import SyncBatchNorm
import glob

import socket
from datetime import datetime
import json
import shutil


torch_ver = torch.__version__[:3]
if torch_ver == "0.3":
    from torch.autograd import Variable


def save_checkpoint(state, directory, is_best, filename="checkpoint.pth.tar"):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + "model_best.pth.tar")


class Trainer:
    def __init__(self, args, directory):
        self.directory = directory
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.log_dir_name = os.path.join(
            self.directory, current_time + "_" + socket.gethostname()
        )

        self.writer = SummaryWriter(log_dir=self.log_dir_name)
        self.writer.add_scalar("train_step/loss", 0, 0)
        self.writer.add_scalar("train_step/mIoU", 0, 0)
        self.writer.add_scalar("train_step/pixAcc", 0, 0)
        self.writer.add_scalar("train_epoch/loss", 0, 0)
        self.writer.add_scalar("train_epoch/mIoU", 0, 0)
        self.writer.add_scalar("train_epoch/pixAcc", 0, 0)
        self.writer.add_scalar("val_epoch/mIoU", 0, 0)
        self.writer.add_scalar("val_epoch/pixAcc", 0, 0)
        self.writer.add_scalar("val_epoch/loss", 0, 0)

        self.args = args
        # data transforms
        if args.use_mapillary_norms:
            im_mean = [0.41738699, 0.45732192, 0.46886091]
            im_std = [0.25685097, 0.26509955, 0.29067996]
        else:
            im_mean = [0.485, 0.456, 0.406]
            im_std = [0.229, 0.224, 0.225]

        input_transform = transform.Compose(
            [transform.ToTensor(), transform.Normalize(im_mean, im_std)]
        )

        # dataset
        data_kwargs = {
            "transform": input_transform,
            "base_size": args.base_size,
            "crop_size": args.crop_size,
            "root": args.data_folder,
        }
        trainset = get_segmentation_dataset(
            args.dataset, split="train", mode="train", **data_kwargs
        )
        testset = get_segmentation_dataset(
            args.dataset, split="val", mode="val", **data_kwargs
        )

        # dataloader
        kwargs = {"num_workers": args.workers, "pin_memory": False} if args.cuda else {}
        self.trainloader = data.DataLoader(
            trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs
        )
        self.valloader = data.DataLoader(
            testset,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle=False,
            **kwargs,
        )
        self.nclass = trainset.num_class

        # model

        model = get_segmentation_model(
            args.model,
            dataset=args.dataset,
            backbone=args.backbone,
            aux=args.aux,
            se_loss=args.se_loss,
            norm_layer=SyncBatchNorm,
            base_size=args.base_size,
            crop_size=args.crop_size,
            **args.custom_dict,
        )

        # count parameter number
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total number of parameters: %d" % pytorch_total_params)

        # optimizer using different LR
        params_list = [{"params": model.pretrained.parameters(), "lr": args.lr}]
        if hasattr(model, "head"):
            if args.diflr:
                params_list.append(
                    {"params": model.head.parameters(), "lr": args.lr * 10}
                )
            else:
                params_list.append({"params": model.head.parameters(), "lr": args.lr})
        if hasattr(model, "auxlayer"):
            if args.diflr:
                params_list.append(
                    {"params": model.auxlayer.parameters(), "lr": args.lr * 10}
                )
            else:
                params_list.append(
                    {"params": model.auxlayer.parameters(), "lr": args.lr}
                )

        optimizer = None
        if args.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params_list,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                params_list, lr=args.lr, weight_decay=args.weight_decay
            )

        if args.swa:
            import torchcontrib  # Todo: move to top after updating image on ngc

            optimizer = torchcontrib.optim.SWA(
                optimizer, swa_start=0, swa_freq=len(self.trainloader), swa_lr=0.05
            )

        self.model, self.optimizer = model, optimizer

        # criterions
        self.losses = {}

        if hasattr(self.model, "num_outputs"):
            num_outputs = self.model.num_outputs
        else:
            num_outputs = 1

        if hasattr(self.model, "aux_indexes"):
            aux_indexes = self.model.aux_indexes
        else:
            aux_indexes = []

        if ~self.args.no_cross:
            self.losses["seg"] = {
                "loss": SegmentationLosses(
                    se_loss=args.se_loss,
                    aux=args.aux,
                    nclass=self.nclass,
                    num_outputs=num_outputs,
                    aux_indexes=aux_indexes,
                ),
                "weight": 1.0,
            }

        if self.args.label_relaxed_loss:
            self.losses["seg"] = {
                "loss": ClassBalancedSegmentationLossesWithLabelRelaxation(
                    se_loss=args.se_loss,
                    aux=args.aux,
                    nclass=self.nclass,
                    num_outputs=num_outputs,
                    aux_indexes=aux_indexes,
                    beta=args.class_balanced_beta,
                ),
                "weight": 1.0,
            }

        if self.args.class_balanced_loss:
            self.losses["seg"] = {
                "loss": ClassBalancedSegmentationLosses(
                    se_loss=args.se_loss,
                    aux=args.aux,
                    nclass=self.nclass,
                    num_outputs=num_outputs,
                    aux_indexes=aux_indexes,
                    beta=args.class_balanced_beta,
                ),
                "weight": 1.0,
            }

        if self.args.soft_dice_loss:
            self.losses["soft_dice"] = {
                "loss": SoftDiceLoss(self.nclass),
                "weight": self.args.dice_gamma,
            }

        if self.args.focal_loss:
            self.losses["focal"] = {
                "loss": FocalLoss(),
                "weight": self.args.focal_gamma,
            }

        if self.args.ohem_loss:
            self.losses["ohem"] = {
                "loss": OHEMSegmentationLosses(
                    se_loss=args.se_loss, aux=args.aux, nclass=self.nclass
                ),
                "weight": 1.0,
            }

        assert len(self.losses) > 0, "There should be at least one loss added"

        # using cuda
        if args.cuda and args.multiple_GPUs:
            self.model = DataParallelModel(self.model).cuda()
            for curr_loss in self.losses.keys():
                self.losses[curr_loss]["loss"] = DataParallelCriterion(
                    self.losses[curr_loss]["loss"]
                ).cuda()

        elif args.cuda and not args.multiple_GPUs:
            self.model = DataParallelModel(self.model).cuda()

            for curr_loss in self.losses.keys():
                self.losses[curr_loss]["loss"] = self.losses[curr_loss]["loss"].cuda()

        self.best_pred = 0.0

        if args.resume_after_suspend:
            if os.path.exists(self.directory):
                files = glob.glob(os.path.join(self.directory, "checkpoint*.pth.tar"))
                if len(files) > 0:
                    files = sorted(files)
                    args.resume = files[-1]
                    args.ft = False

        # resuming checkpoint
        if args.resume is not None and len(args.resume) > 0:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            if args.cuda:
                # load weights for the same model
                # model and checkpoint have different strucutures
                pretrained_dict = checkpoint["state_dict"]
                model_dict = self.model.module.state_dict()

                for name, param in pretrained_dict.items():
                    if name not in model_dict:
                        print("No weights founds for %s layer in loaded model." % name)
                        continue
                    if isinstance(param, Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    model_dict[name].copy_(param)

            else:
                self.model.load_state_dict(checkpoint["state_dict"])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_pred = checkpoint["best_pred"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )

        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
            self.best_pred = 0.0

        # lr scheduler
        self.scheduler = utils.LR_Scheduler(
            args.lr_scheduler,
            args.lr,
            args.epochs,
            len(self.trainloader),
            lr_step=args.lr_step,
        )

    def training(self, epoch):
        # progressive resizing
        if self.args.epochs_per_resize is not None:
            if epoch < self.args.epochs_per_resize:
                self.trainloader.dataset.base_size = self.args.base_size // 8
                self.trainloader.dataset.crop_size = self.args.crop_size // 8
            elif epoch < self.args.epochs_per_resize * 2:
                self.trainloader.dataset.base_size = self.args.base_size // 4
                self.trainloader.dataset.crop_size = self.args.crop_size // 4
            elif epoch < self.args.epochs_per_resize * 3:
                self.trainloader.dataset.base_size = self.args.base_size // 2
                self.trainloader.dataset.crop_size = self.args.crop_size // 2
            else:
                self.trainloader.dataset.base_size = self.args.base_size
                self.trainloader.dataset.crop_size = self.args.crop_size

        # mini-batch loop
        train_loss = 0.0
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.trainloader)
        for i, (images, targets) in enumerate(tbar):
            self.model.train()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            if torch_ver == "0.3":
                images = Variable(images)
                targets = Variable(targets)

            if self.args.multiple_GPUs:
                outputs = self.model(images)
                loss = 0
                for curr_loss in self.losses.keys():
                    loss += self.losses[curr_loss]["weight"] * self.losses[curr_loss][
                        "loss"
                    ](outputs, targets)
            else:
                outputs = self.model(images.cuda())

                loss = 0
                for curr_loss in self.losses.keys():
                    loss += self.losses[curr_loss]["weight"] * self.losses[curr_loss][
                        "loss"
                    ](*outputs, targets.cuda())

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                correct, labeled, inter, union = self.eval_batch(
                    self.model, images, targets
                )

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            tbar.set_description(
                "Train loss: %.3f   mIoU: %.3f" % (train_loss / (i + 1), mIoU)
            )

            if i % 100 == 0:
                self.writer.add_scalar(
                    "train_step/loss",
                    train_loss / (i + 1),
                    epoch * len(self.trainloader) + i,
                )
                self.writer.add_scalar(
                    "train_step/mIoU", mIoU, epoch * len(self.trainloader) + i
                )
                self.writer.add_scalar(
                    "train_step/pixAcc", pixAcc, epoch * len(self.trainloader) + i
                )

        if self.args.save_interval > 0:
            if (epoch + 1) % self.args.save_interval == 0:
                # save checkpoint every epoch
                is_best = False
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.module.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_pred": self.best_pred,
                    },
                    self.directory,
                    is_best,
                    filename="checkpoint_%03d.pth.tar" % (epoch + 1),
                )

        self.writer.add_scalar("train_epoch/loss", train_loss / (i + 1), epoch + 1)
        self.writer.add_scalar("train_epoch/mIoU", mIoU, epoch + 1)
        self.writer.add_scalar("train_epoch/pixAcc", pixAcc, epoch + 1)

        # Fast test during the training

    def eval_batch(self, model, images, targets, return_loss=False):
        if self.args.multiple_GPUs:
            outputs = model(images)
            loss = 0
            for curr_loss in self.losses.keys():
                loss += self.losses[curr_loss]["weight"] * self.losses[curr_loss][
                    "loss"
                ](outputs, targets)

            outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]

        else:
            outputs = model(images.cuda())
            loss = 0
            for curr_loss in self.losses.keys():
                loss += self.losses[curr_loss]["weight"] * self.losses[curr_loss][
                    "loss"
                ](outputs[0], targets.cuda())
            # outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]

        targets = targets.cuda()
        correct, labeled = utils.batch_pix_accuracy(pred.data, targets)
        inter, union = utils.batch_intersection_union(pred.data, targets, self.nclass)
        if return_loss:
            return correct, labeled, inter, union, loss
        else:
            return correct, labeled, inter, union

    def validation(self, epoch):
        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc="\r")

        for i, (images, targets) in enumerate(tbar):
            if torch_ver == "0.3":
                images = Variable(images, volatile=True)
                correct, labeled, inter, union, loss = self.eval_batch(
                    self.model, images, targets, return_loss=True
                )
            else:
                with torch.no_grad():
                    correct, labeled, inter, union, loss = self.eval_batch(
                        self.model, images, targets, return_loss=True
                    )

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            total_loss += loss
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            tbar.set_description(
                "Val   loss: %.3f   mIoU: %.3f" % (total_loss / (i + 1), mIoU)
            )

        self.writer.add_scalar("val_epoch/mIoU", mIoU, epoch + 1)
        self.writer.add_scalar("val_epoch/pixAcc", pixAcc, epoch + 1)
        self.writer.add_scalar("val_epoch/loss", (total_loss) / (i + 1), epoch + 1)
        self.writer.flush()

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
                self.directory,
                is_best,
            )


def main():
    args = Options().parse()
    args.custom_dict = json.loads(args.custom_dict)
    print(args)
    torch.manual_seed(args.seed)

    checkname = args.checkname + "_" + args.backbone + "_" + "%04d" % args.batch_size
    print(f"checkname is {checkname}")
    directory = os.path.join(
        args.save_dir, "%s/%s/%s/" % (args.dataset, args.model, checkname)
    )

    trainer = Trainer(args, directory)
    print("Starting Epoch:", args.start_epoch)
    print("Total Epochs:", args.epochs)

    with EnergyMeter(dir=directory) if args.em else ExitStack():
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if not args.no_val:
                trainer.validation(epoch)

    if args.swa:
        trainer.optimizer.swap_swa_sgd()
        trainer.optimizer.bn_update(trainer.train_loader, trainer.model)
        if not args.no_val:
            print("Evaluation after swa swap ......")
            trainer.validation(epoch)


if __name__ == "__main__":
    main()
