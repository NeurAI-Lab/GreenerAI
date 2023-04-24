###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import argparse
import torch
import uuid


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="PyTorch \
            Segmentation"
        )
        # model and dataset
        parser.add_argument(
            "--model",
            type=str,
            default="shelfnet",
            help="model name (default: shelfnet)",
        )
        parser.add_argument(
            "--em", action="store_true", help="measure the energy during training"
        )
        parser.add_argument(
            "--diflr",
            action="store_true",
            default=False,
            help="use different lr for head and backbone if set as True",
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="resnet101",
            help="backbone name (default: resnet101)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="mapillary",
            help="dataset name (default: pascal_voc)",
        )
        parser.add_argument(
            "--data-folder",
            type=str,
            default=os.path.join("/volumes1/datasets/Mapillary_v1.1"),
            help="training dataset folder (default: \
                            $(HOME)/data)",
        )
        parser.add_argument(
            "--workers", type=int, default=20, metavar="N", help="dataloader threads"
        )
        parser.add_argument(
            "--base-size", type=int, default=1024, help="base image size"  # 608  #1024
        )
        parser.add_argument(
            "--crop-size", type=int, default=768, help="crop image size"  # 512  #768
        )

        # training hyper params
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=None,
            metavar="N",
            help="number of epochs to train (default: auto)",
        )
        parser.add_argument(
            "--epochs_per_resize",
            type=int,
            default=None,
            metavar="N",
            help=(
                "if specified, use progressive resizing: ",
                "train for N epochs at 1/8 size, "
                "then for N epochs at 1/4 size, then for N epochs at 1/2 size, "
                "then continue at full resolution",
            ),
        )
        parser.add_argument(
            "--start_epoch",
            type=int,
            default=0,
            metavar="N",
            help="start epochs (default:0)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            metavar="N",
            help="input batch size for \
                            training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=None,
            metavar="N",
            help="input batch size for \
                            testing (default: same as batch size)",
        )

        # dice_loss options
        parser.add_argument(
            "--no-cross", action="store_true", default=False, help="use focal_loss "
        )

        parser.add_argument(
            "--cross-gamma", type=float, default=1.0, help="dice_loss gamma"
        )

        parser.add_argument(
            "--soft-dice-loss",
            action="store_true",
            default=False,
            help="use dice_loss ",
        )

        parser.add_argument(
            "--dice-gamma", type=float, default=1.0, help="dice_loss gamma"
        )

        # focal_loss options
        parser.add_argument(
            "--focal-loss", action="store_true", default=False, help="use focal_loss "
        )

        parser.add_argument(
            "--focal-gamma", type=float, default=1.0, help="focal_loss gamma"
        )

        parser.add_argument(
            "--class-balanced-loss",
            action="store_true",
            default=False,
            help="use class balanced loss",
        )

        parser.add_argument(
            "--class-balanced-beta", type=float, default=0.999, help="focal_loss gamma"
        )

        parser.add_argument(
            "--label-relaxed-loss",
            action="store_true",
            default=False,
            help="label relaxed loss",
        )

        parser.add_argument(
            "--ohem-loss", action="store_true", default=False, help="label relaxed loss"
        )

        # optimizer params
        parser.add_argument(
            "--lr",
            type=float,
            default=None,
            metavar="LR",
            help="learning rate (default: auto)",
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="sgd",
            metavar="OPTIMIZER",
            help="learning rate (default: auto)",
        )
        parser.add_argument(
            "--lr-scheduler",
            type=str,
            default="poly",
            help="learning rate scheduler (default: poly)",
        )
        parser.add_argument(
            "--lr-step", type=int, default=35, help="steps to decay learing rate by 0.1"
        )
        parser.add_argument(
            "--momentum",
            type=float,
            default=0.9,
            metavar="M",
            help="momentum (default: 0.9)",
        )
        parser.add_argument(
            "--weight-decay",
            type=float,
            default=1e-4,
            metavar="M",
            help="w-decay (default: 1e-4)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--multiple-GPUs", default=False, help="train with multiple GPUs"
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        # checking point
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="put the path to resuming file if needed",
        )

        parser.add_argument(
            "--resume-after-suspend",
            default=True,
            help="skip validation during training",
        )
        parser.add_argument(
            "--checkname",
            type=str,
            default="base_model_" + str(uuid.uuid4()),
            help="set the checkpoint name",
        )
        parser.add_argument(
            "--model-zoo", type=str, default=None, help="evaluating on model zoo model"
        )
        # finetuning pre-trained models
        parser.add_argument(
            "--ft",
            action="store_true",
            default=False,
            help="finetuning on a different dataset",
        )
        parser.add_argument(
            "--pre-class",
            type=int,
            default=None,
            help="num of pre-trained classes \
                            (default: None)",
        )
        parser.add_argument(
            "--pretrained",
            action="store_true",
            default=False,
            help="use pretrained backbone",
        )
        # evaluation option
        parser.add_argument(
            "--ema", action="store_true", default=False, help="using EMA evaluation"
        )
        parser.add_argument(
            "--eval", action="store_true", default=False, help="evaluating mIoU"
        )
        parser.add_argument(
            "--no-val",
            action="store_true",
            default=False,
            help="skip validation during training",
        )
        # test option
        parser.add_argument(
            "--test-folder", type=str, default=None, help="path to test image folder"
        )
        parser.add_argument(
            "--save-interval", type=int, default=10, help="save_interval"
        )

        parser.add_argument(
            "--save-dir",
            type=str,
            default="runs/",
            help="where model checkpoint and summary should save.",
        )

        parser.add_argument(
            "--norm-layer", type=str, default="bn", help="path to test image folder"
        )
        parser.add_argument(
            "--swa",
            default=False,
            action="store_true",
            help="path to test image folder",
        )

        parser.add_argument("--custom-dict", default="{}", type=str)

        parser.add_argument("--fusion-mode", default="mean", type=str)

        parser.add_argument(
            "-s",
            "--scales",
            nargs="+",
            help="<Required> Set flag",
            type=float,
            dest="scales",
            default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        )

        parser.add_argument(
            "--use-mapillary-norms",
            default=False,
            action="store_true",
            help="path to test image folder",
        )

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                "pascal_voc": 50,
                "pascal_aug": 50,
                "pcontext": 80,
                "ade20k": 160,
                "coco": 30,
                "citys": 80,
                "citys_coarse": 60,
                "nie": 150,
                "mapillary": 300,
                "mapillary_v2": 300,
                "mapillary_old": 300,
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.batch_size is None:
            args.batch_size = 1 * torch.cuda.device_count()
        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size
        if args.lr is None:
            lrs = {
                "pascal_voc": 0.0001,
                "pascal_aug": 0.001,
                "pcontext": 0.001,
                "ade20k": 0.01,
                "coco": 0.01,
                "citys": 0.001,
                "citys_coarse": 0.01,
                "nie": 0.001,
                "mapillary": 0.001,
                "mapillary_v2": 0.001,
                "mapillary_old": 0.001,
            }
            args.lr = lrs[args.dataset.lower()]
        return args
