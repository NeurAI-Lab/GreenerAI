###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter

from encoding_custom import backbones
from encoding.utils import batch_pix_accuracy, batch_intersection_union
from encoding_custom.models.fpn import FPN

from encoding_custom.in_place_abn import models as inplace_models
import os


up_kwargs = {"mode": "bilinear", "align_corners": True}

__all__ = ["BaseNet", "MultiEvalModule"]


class BaseNet(nn.Module):
    def __init__(
        self,
        nclass,
        backbone,
        aux,
        se_loss,
        dilated=False,
        norm_layer=None,
        base_size=576,
        crop_size=608,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        root="~/.encoding/models",
        dilate_only_last_layer=False,
        pretrained=True,
        multi_grid=True,
        multi_dilation=[4, 8, 16],
    ):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        self.backbone = backbone
        # copying modules from pretrained models
        if backbone.startswith("resnet"):
            self.pretrained = {
                "resnet18": backbones.resnet18,
                "resnet34": backbones.resnet34,
                "resnet50": backbones.resnet50,
                "resnet101": backbones.resnet101,
                "resnet152": backbones.resnet152,
            }[backbone](
                pretrained=pretrained,
                dilated=dilated,
                norm_layer=norm_layer,
                root=root,
                dilate_only_last_layer=dilate_only_last_layer,
            )
        elif backbone == "se_resnet101":
            self.pretrained = backbones.se_resnet101()
        elif backbone == "se_resnext101":
            self.pretrained = backbones.se_resnext101_32x4d()
        elif backbone == "se_resnet152":
            self.pretrained = backbones.se_resnet152()
        elif backbone == "inception_v4":
            self.pretrained = backbones.inceptionv4(pretrained="imagenet", root=root)

        elif backbone == "fpn_resnet18":
            self.pretrained = FPN(
                backbones.resnet18(
                    pretrained=pretrained, dilated=False, add_additional_layers=False
                ),
                has_block_expansion=False,
            )

        elif backbone == "fpn_resnet50":
            self.pretrained = FPN(
                backbones.resnet50(
                    pretrained=pretrained, dilated=False, add_additional_layers=False
                )
            )

        elif backbone == "fpn_resnet101":
            self.pretrained = FPN(
                backbones.resnet101(
                    pretrained=pretrained, dilated=False, add_additional_layers=False
                )
            )

        elif backbone == "fpn_se_resnet101":
            self.pretrained = FPN(backbones.se_resnet101())

        elif backbone == "fpn_sge_resnet101":
            self.pretrained = FPN(backbones.sge_resnet101())

        elif backbone == "sge_resnet101":
            self.pretrained = backbones.sge_resnet101()

        elif backbone == "mobile_net_v3":
            self.pretrained = backbones.mobilenetv3_large()

        elif backbone == "dilated_resnet101":
            self.pretrained = backbones.dilated_resnet101(
                pretrained=True,
                dilated=dilated,
                norm_layer=norm_layer,
                root=root,
                multi_grid=multi_grid,
                multi_dilation=multi_dilation,
            )

        elif backbone == "fpn_dilated_resnet101":
            self.pretrained = FPN(
                backbones.dilated_resnet101(
                    pretrained=True,
                    dilated=dilated,
                    norm_layer=norm_layer,
                    root=root,
                    multi_grid=multi_grid,
                    multi_dilation=multi_dilation,
                )
            )

        elif backbone == "inplace_resnet101":
            from encoding_custom.in_place_abn.modules import InPlaceABN, InPlaceABNSync
            from functools import partial

            norm_layer_inplace = partial(
                InPlaceABNSync, activation="leaky_relu", slope=0.01
            )

            self.pretrained = inplace_models.__dict__["net_resnet101"](
                norm_act=norm_layer_inplace, dilation=1, keep_outputs=True
            )
            if pretrained:
                pretrained_path = "/input/pretrained/modified_inplace_resnet101.pth.tar"
                assert os.path.exists(pretrained_path)
                data = torch.load(pretrained_path)
                self.pretrained.load_state_dict(data["state_dict"])

        elif backbone == "again_resnet101":
            from encoding_custom.backbones.models_lpf.resnet import (
                resnet101 as again_resnet101,
            )

            self.pretrained = again_resnet101(
                pretrained=pretrained, norm_layer=norm_layer, filter_size=3
            )

        elif "efficientnet" in backbone:
            self.pretrained = backbones.efficientnet_pytorch.EfficientNet.from_pretrained(
                backbone
            )

        elif backbone is None:
            self.pretrained = nn.Sequential()
        else:
            raise RuntimeError("unknown backbone: {}".format(backbone))

        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        if "inplace" in self.backbone:
            _, c1, c2, c3, c4 = self.pretrained(x)
        elif "efficientnet" in self.backbone:
            c1, c2, c3, c4 = self.pretrained(x)
        elif isinstance(self.pretrained, FPN):
            c1, c2, c3, c4 = self.pretrained(x)
        elif isinstance(self.pretrained, backbones.MobileNetV3):
            out = [x]
            for feat in self.pretrained.features:
                out.append(feat(out[-1]))
            return tuple(out[1:])
        else:
            if hasattr(self.pretrained, "layer0"):
                x = self.pretrained.layer0(x)
            else:
                x = self.pretrained.conv1(x)
                x = self.pretrained.bn1(x)
                x = self.pretrained.relu(x)
                x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union


class BaseNet_v2(nn.Module):
    def __init__(
        self,
        nclass,
        backbone,
        aux,
        se_loss,
        dilated=True,
        norm_layer=None,
        base_size=576,
        crop_size=608,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        root="~/.encoding/models",
    ):
        super(BaseNet_v2, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        # copying modules from pretrained models
        if backbone == "resnet50":
            pretrained = resnet.resnet50(
                pretrained=True, dilated=dilated, norm_layer=norm_layer, root=root
            )
        elif backbone == "resnet101":
            pretrained = resnet.resnet101(
                pretrained=True, dilated=dilated, norm_layer=norm_layer, root=root
            )
        elif backbone == "resnet152":
            pretrained = resnet.resnet152(
                pretrained=True, dilated=dilated, norm_layer=norm_layer, root=root
            )
        else:
            raise RuntimeError("unknown backbone: {}".format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs
        self.pretrained = ExtractPretrained(pretrained)

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        return c1, c2, c3

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union


class ExtractPretrained(nn.Module):
    def __init__(self, pretrained):
        super(ExtractPretrained, self).__init__()
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3


class MultiEvalModule(DataParallel):
    """Multi-size Segmentation Eavluator"""

    def __init__(
        self,
        module,
        nclass,
        device_ids=None,
        flip=True,
        scales=[0.5, 0.75, 1.0, 1.25, 1.50, 1.75],
    ):
        super(MultiEvalModule, self).__init__(module, device_ids)
        self.nclass = nclass
        self.base_size = module.base_size
        self.crop_size = module.crop_size
        self.scales = scales
        self.flip = flip
        print(
            "MultiEvalModule: base_size {}, crop_size {}".format(
                self.base_size, self.crop_size
            )
        )

    def parallel_forward(self, inputs, **kwargs):
        """Multi-GPU Mult-size Evaluation

        Args:
            inputs: list of Tensors
        """
        inputs = [
            (input.unsqueeze(0).cuda(device),)
            for input, device in zip(inputs, self.device_ids)
        ]
        replicas = self.replicate(self, self.device_ids[: len(inputs)])
        kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return outputs

    def forward(self, image):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        batch, _, h, w = image.size()
        assert batch == 1
        stride_rate = 2.0 / 3.0
        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch, self.nclass, h, w).zero_().cuda()

        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            # resize image to current size
            cur_img = resize_image(image, height, width, **self.module._up_kwargs)
            if long_size <= crop_size:
                pad_img = pad_image(
                    cur_img, self.module.mean, self.module.std, crop_size
                )
                outputs = module_inference(self.module, pad_img, self.flip)
                outputs = crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = pad_image(
                        cur_img, self.module.mean, self.module.std, crop_size
                    )
                else:
                    pad_img = cur_img
                _, _, ph, pw = pad_img.size()
                assert ph >= height and pw >= width
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
                with torch.cuda.device_of(image):
                    outputs = (
                        image.new().resize_(batch, self.nclass, ph, pw).zero_().cuda()
                    )
                    count_norm = image.new().resize_(batch, 1, ph, pw).zero_().cuda()
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = pad_image(
                            crop_img, self.module.mean, self.module.std, crop_size
                        )
                        output = module_inference(self.module, pad_crop_img, self.flip)
                        outputs[:, :, h0:h1, w0:w1] += crop_image(
                            output, 0, h1 - h0, 0, w1 - w0
                        )
                        count_norm[:, :, h0:h1, w0:w1] += 1
                assert (count_norm == 0).sum() == 0
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]

            score = resize_image(outputs, h, w, **self.module._up_kwargs)
            scores += score

        return scores


def module_inference(module, image, flip=True):
    output = module.evaluate(image)
    if flip:
        fimg = flip_image(image)
        foutput = module.evaluate(fimg)
        output += flip_image(foutput)
    return output.exp()


def resize_image(img, h, w, **up_kwargs):
    return F.upsample(img, (h, w), **up_kwargs)


def pad_image(img, mean, std, crop_size):
    b, c, h, w = img.size()
    assert c == 3
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b, c, h + padh, w + padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:, i, :, :] = F.pad(
            img[:, i, :, :], (0, padw, 0, padh), value=pad_values[i]
        )
    assert img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size
    return img_pad


def crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]


def flip_image(img):
    assert img.dim() == 4
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3) - 1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)
