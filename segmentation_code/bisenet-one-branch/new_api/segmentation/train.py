import sys

sys.path.insert(0, "../../")

import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import numpy as np

# from coco_utils import get_coco
import utils
import glob
import uuid
from tensorboardX import SummaryWriter
import socket

from encoding_custom.datasets import get_segmentation_dataset
import torchvision.transforms as transform

from encoding_custom.models import get_segmentation_model
from encoding_custom.in_place_abn.modules import InPlaceABN, InPlaceABNSync
from encoding_custom.in_place_abn.modules.sync_sn_layer import SyncSwitchableNorm2d
from encoding_custom.in_place_abn.modules.switch_norm import SwitchNorm2d
from encoding.nn import SyncBatchNorm
import json
from functools import partial


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=-1)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            output = output["out"]

            metric_logger.update(loss=loss.item())
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat, metric_logger


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    lr_scheduler,
    device,
    epoch,
    print_freq,
    num_classes,
    writer,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = "Epoch: [{}]".format(epoch)
    confmat = utils.ConfusionMatrix(num_classes)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        confmat.update(target.flatten(), output["out"].argmax(1).flatten())
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    confmat.reduce_from_all_processes()
    acc_global, mean_iou, _ = confmat.get_metrics()
    writer.add_scalar("train/loss_epoch", metric_logger.loss.avg, epoch + 1)
    writer.add_scalar("train/mean_iou", mean_iou, epoch)
    writer.add_scalar("train/acc", acc_global, epoch)

    print(metric_logger)
    print(confmat)


def main(args):
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6666"
    args.checkname = (
        args.checkname + "_" + args.backbone + "_" + "%04d" % args.batch_size
    )
    utils.mkdir(
        os.path.join(
            args.output_dir, "%s/%s/%s" % (args.dataset, args.model, args.checkname)
        )
    )

    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    writer = SummaryWriter(
        log_dir=os.path.join(
            args.output_dir,
            current_time
            + "_"
            + socket.gethostname()
            + "_"
            + str(args.dataset)
            + "_"
            + str(args.backbone)
            + "_b"
            + str(args.batch_size)
            + "_"
            + args.checkname,
        )
    )

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl")
    # utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    input_transform = transform.Compose(
        [
            transform.ToTensor(),
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # dataset
    data_kwargs = {
        "transform": input_transform,
        "base_size": args.base_size,
        "crop_size": args.crop_size,
        "root": args.data_folder,
    }
    dataset = get_segmentation_dataset(
        args.dataset, split="train", mode="train", **data_kwargs
    )
    dataset_test = get_segmentation_dataset(
        args.dataset, split="val", mode="val", **data_kwargs
    )
    num_classes = dataset.NUM_CLASS

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers
    )

    if args.norm_layer == "bn":
        norm_layer = SyncBatchNorm
    elif args.norm_layer == "inplace_relu":
        norm_layer = partial(InPlaceABNSync, activation="leaky_relu", slope=0.01)
    elif args.norm_layer == "sn":
        norm_layer = SyncSwitchableNorm2d if args.distributed else SwitchNorm2d
    else:
        raise ("norm layer not found")

    model = get_segmentation_model(
        args.model,
        dataset=args.dataset,
        backbone=args.backbone,
        aux=args.aux,
        se_loss=args.se_loss,
        norm_layer=norm_layer,
        base_size=args.base_size,
        crop_size=args.crop_size,
        **args.custom_dict
    )

    if args.use_model_wrapper:
        model = utils.SSM(model)

    model.to(device)

    start_epoch = 0

    if args.resume is None and args.resume_after_suspend:
        directory = os.path.join(
            args.output_dir, "%s/%s/%s/" % (args.dataset, args.model, args.checkname)
        )
        if os.path.exists(directory):
            files = glob.glob(os.path.join(directory, "*.pth"))
            if len(files) > 0:
                files = sorted(files)
                args.resume = files[-1]

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if args.resume_after_suspend and "epoch" in checkpoint.keys():
            start_epoch = checkpoint["epoch"]

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module

    if args.test_only:
        confmat = evaluate(
            model, data_loader_test, device=device, num_classes=num_classes
        )
        print(confmat)
        return

    if args.use_model_wrapper:
        model_without_ddp = model_without_ddp.model

    params_to_optimize = [
        {
            "params": [
                p for p in model_without_ddp.pretrained.parameters() if p.requires_grad
            ]
        },
        {"params": [p for p in model_without_ddp.head.parameters() if p.requires_grad]},
    ]
    if args.aux:
        params = [p for p in model_without_ddp.auxlayer.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9
    )

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            lr_scheduler,
            device,
            epoch,
            args.print_freq,
            num_classes,
            writer,
        )
        val_confmat, val_metric_logger = evaluate(
            model, data_loader_test, device=device, num_classes=num_classes
        )

        acc_global, mean_iou, _ = val_confmat.get_metrics()
        writer.add_scalar("val/loss_epoch", val_metric_logger.loss.avg, epoch + 1)
        writer.add_scalar("val/mean_iou", mean_iou, epoch)
        writer.add_scalar("val/acc", acc_global, epoch)

        print(val_confmat)
        print(val_metric_logger)

        utils.save_on_master(
            {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "args": args,
            },
            os.path.join(
                args.output_dir,
                "%s/%s/%s/" % (args.dataset, args.model, args.checkname),
                "model_{}.pth".format(epoch),
            ),
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")

    parser.add_argument("--dataset", default="citys", help="dataset")
    parser.add_argument("--model", default="fcn_resnet101", help="model")
    parser.add_argument("--aux", action="store_true", help="auxiliar loss")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=16,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./runs", help="path where to save")
    parser.add_argument("--resume", default=None, help="resume from checkpoint")

    parser.add_argument(
        "--test-only", dest="test_only", help="Only test the model", action="store_true"
    )

    ####
    parser.add_argument(
        "--distributed", action="store_true", help="go multi gpu", default=False
    )

    parser.add_argument(
        "--data-folder",
        type=str,
        default=os.path.join("/volumes1/datasets/city_scape_v2"),
        help="training dataset folder (default: \
                        $(HOME)/data)",
    )
    parser.add_argument(
        "--checkname",
        type=str,
        default="init_model_" + str(uuid.uuid4()),
        help="set the checkpoint name",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet101",
        help="backbone name (default: resnet101)",
    )

    parser.add_argument(
        "--resume-after-suspend", default=True, help="skip validation during training"
    )

    parser.add_argument(
        "--base-size", type=int, default=1024, help="base image size"  # 608  #1024
    )
    parser.add_argument(
        "--crop-size", type=int, default=768, help="crop image size"  # 512  #768
    )

    parser.add_argument(
        "--se-loss",
        action="store_true",
        default=False,
        help="Semantic Encoding Loss SE-loss",
    )

    parser.add_argument("--custom-dict", default="{}", type=str)
    ####

    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    parser.add_argument("--use-model-wrapper", default=False, action="store_true")

    parser.add_argument(
        "--norm-layer",
        type=str,
        default="inplace_relu",
        help="path to test image folder",
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )

    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.custom_dict = json.loads(args.custom_dict)

    main(args)
