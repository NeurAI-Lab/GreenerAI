"""Use it like this:
python test_multi_fusion.py --model bisenet --backbone resnet18 \
        --resume citys/bisenet/pretrain_resnet18_0016/checkpoint_1000.pth.tar \
        --scales 1 --eval --dataset citys \
        --data-folder /data/input/datasets/cityscape_processed/

Notes:
    - Doesnt work with multiple gpus! need to set parallel model
"""

import sys

sys.path.insert(0, "../../")

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding_custom.utils.metrics_custom import SegmentationMetricCustom
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding_custom.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, MultiEvalModule
from encoding_custom.models import get_segmentation_model
from experiments.segmentation.option import Options
import torch.nn.functional as functional
import torch.nn as nn
from collections import OrderedDict
from encoding_custom.in_place_abn.modules.switch_norm import SwitchNorm2d
from encoding_custom.in_place_abn.modules.sync_sn_layer import SyncSwitchableNorm2d

from functools import partial
from encoding_custom.in_place_abn.modules import InPlaceABN, InPlaceABNSync
from PIL import Image
from encoding_custom.utils.energy_meter import EnergyMeter


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]


class SegmentationModule(nn.Module):
    _IGNORE_INDEX = -1  # hardcoded to cityscape!

    class _MeanFusion:
        def __init__(self, x, classes):
            self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.counter = 0

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            self.counter += 1
            self.buffer.add_((probs - self.buffer) / self.counter)

        def output(self):
            probs, cls = self.buffer.max(1)
            return probs, cls

    class _VotingFusion:
        def __init__(self, x, classes):
            self.votes = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.probs = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            probs, cls = probs.max(1, keepdim=True)

            self.votes.scatter_add_(1, cls, self.votes.new_ones(cls.size()))
            self.probs.scatter_add_(1, cls, probs)

        def output(self):
            cls, idx = self.votes.max(1, keepdim=True)
            probs = self.probs / self.votes.clamp(min=1)
            probs = probs.gather(1, idx)
            return probs.squeeze(1), cls.squeeze(1)

    class _MaxFusion:
        def __init__(self, x, _):
            self.buffer_cls = x.new_zeros(
                x.size(0), x.size(2), x.size(3), dtype=torch.long
            )
            self.buffer_prob = x.new_zeros(x.size(0), x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            max_prob, max_cls = probs.max(1)

            replace_idx = max_prob > self.buffer_prob
            self.buffer_cls[replace_idx] = max_cls[replace_idx]
            self.buffer_prob[replace_idx] = max_prob[replace_idx]

        def output(self):
            return self.buffer_prob, self.buffer_cls

    def __init__(self, model, classes, fusion_mode="mean"):
        super(SegmentationModule, self).__init__()
        self.model = model
        self.classes = classes
        if fusion_mode == "mean":
            self.fusion_cls = SegmentationModule._MeanFusion
        elif fusion_mode == "voting":
            self.fusion_cls = SegmentationModule._VotingFusion
        elif fusion_mode == "max":
            self.fusion_cls = SegmentationModule._MaxFusion

    def _network(self, x, scale):
        if scale != 1:
            scaled_size = [round(s * scale) for s in x.shape[-2:]]
            x_up = functional.upsample(x, size=scaled_size, mode="bilinear")
        else:
            x_up = x

        sem_logits = self.model(x_up)

        if isinstance(sem_logits, OrderedDict):
            sem_logits = sem_logits["out"]
        else:
            sem_logits = sem_logits[0]
        del x_up
        return sem_logits

    def forward(self, x, scales, do_flip=False):
        out_size = x.shape[-2:]
        fusion = self.fusion_cls(x, self.classes)

        for scale in scales:
            # Main orientation
            sem_logits = self._network(x, scale)
            sem_logits = functional.upsample(sem_logits, size=out_size, mode="bilinear")
            fusion.update(sem_logits)

            # Flipped orientation
            if do_flip:
                # Main orientation
                sem_logits = self._network(flip(x, -1), scale)
                sem_logits = functional.upsample(
                    sem_logits, size=out_size, mode="bilinear"
                )
                fusion.update(flip(sem_logits, -1))

        return fusion.output()


def test(args):
    # output folder
    outdir = args.save_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
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
    if args.eval:
        testset = get_segmentation_dataset(
            args.dataset,
            split="val",
            mode="testval",
            transform=input_transform,
            root=args.data_folder,
        )
    # elif args.test_val:
    #     testset = get_segmentation_dataset(args.dataset, split='val', mode='test',
    #                                        transform=input_transform, root=args.data_folder)
    else:
        testset = get_segmentation_dataset(
            args.dataset,
            split="test",
            mode="test",
            transform=input_transform,
            root=args.data_folder,
        )
    # dataloader
    loader_kwargs = (
        {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
    )
    test_data = data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=test_batchify_fn,
        **loader_kwargs
    )
    if args.norm_layer == "bn":
        norm_layer = SyncBatchNorm
    elif args.norm_layer == "inplace_relu":
        norm_layer = partial(InPlaceABNSync, activation="leaky_relu", slope=0.01)
    elif args.norm_layer == "sn":
        norm_layer = SyncSwitchableNorm2d if args.distributed else SwitchNorm2d
    else:
        raise ("norm layer not found")
    if args.shift_invariant:
        assert args.model == "bisenet", "Shift Invariance only applicable to BiSeNet"

    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
        # model.base_size = args.base_size
        # model.crop_size = args.crop_size
    else:
        model = get_segmentation_model(
            args.model,
            dataset=args.dataset,
            backbone=args.backbone,
            aux=args.aux,
            se_loss=args.se_loss,
            norm_layer=norm_layer,
            base_size=args.base_size,
            crop_size=args.crop_size,
            anti_aliased=args.shift_invariant,
        )
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        if "state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict"])
        elif "model" in checkpoint.keys():
            model.load_state_dict(checkpoint["model"])
        else:
            raise ("loaded checkpoint has no params key!")

        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )

    # scales = [0.7, 1, 1.2]
    # scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    scales = args.scales
    print("scales:", scales)
    evaluator = SegmentationModule(
        model, testset.num_class, fusion_mode=args.fusion_mode
    ).cuda()
    evaluator.eval()
    metric = SegmentationMetricCustom(testset.num_class)

    tbar = tqdm(test_data)
    for i, (image, dst) in enumerate(tbar):
        image = image[0].unsqueeze(0).cuda()
        if args.eval:
            with torch.no_grad():
                dst = dst[0].unsqueeze(0).cuda().half()
                image = functional.interpolate(image, scale_factor=0.5, align_corners=False, mode='bilinear')
                dst = functional.interpolate(dst.unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0).long()

                _, predicts = evaluator(image, scales)
                metric.update(dst, predicts)
                pixAcc, mIoU = metric.get()
                tbar.set_description("pixAcc: %.4f, mIoU: %.4f" % (pixAcc, mIoU))
        else:
            with torch.no_grad():
                _, outputs = evaluator(image, scales)
                # predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                #             for output in outputs]
                predicts = [testset.make_pred(outputs.cpu().numpy())]
            for predict, impath in zip(predicts, dst):
                mask = Image.fromarray(predict.squeeze().astype("uint8"))
                # mask = utils.get_mask_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + ".png"
                mask.save(os.path.join(outdir, outname))

    return metric.get()


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    # print('fusion mode: mean')
    # args.fusion_mode = 'mean'
    pixAcc, mIoU = test(args)
    print("pixAcc: %.4f, mIoU: %.4f" % (pixAcc, mIoU))

    # print('fusion mode: max')
    # args.fusion_mode = 'max'
    # pixAcc, mIoU = test(args)
    # print('pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))

    # print('fusion mode: voting')
    # args.fusion_mode = 'voting'
    # pixAcc, mIoU = test(args)
    # print('pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
