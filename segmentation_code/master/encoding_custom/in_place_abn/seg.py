from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from encoding_custom.in_place_abn import models as inplace_models
from functools import partial
from encoding_custom.in_place_abn.modules.deeplab import DeeplabV3
from encoding_custom.models.fast_laddernet_se import LadderHead
from encoding_custom.in_place_abn.modules import InPlaceABN, InPlaceABNSync
import os
from encoding_custom.datasets import datasets
from encoding.utils import batch_pix_accuracy, batch_intersection_union
from encoding_custom import backbones

up_kwargs = {"mode": "bilinear", "align_corners": False}

model_dict = {"resnet101": backbones.resnet101, "resnet50": backbones.resnet50}


class SimpleSegmentationModel(nn.Module):
    def __init__(
        self,
        backbone,
        model,
        num_classes,
        aux_classifier=None,
        distributed=False,
        pretrained=True,
        pretrained_path="/input/pretrained/wide_resnet38_deeplab_vistas_modified.pth.tar",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        norm_layer=None,
        dilated=True,
        root="~/.encoding/models",
    ):
        super(SimpleSegmentationModel, self).__init__()

        dilation = (1, 2, 4, 4) if dilated else 1

        if model == "deeplab":
            norm_layer = partial(InPlaceABNSync, activation="leaky_relu", slope=0.01)
            body = inplace_models.__dict__[backbone](
                norm_act=norm_layer, dilation=dilation, keep_outputs=False
            )

            head = DeeplabV3(4096, 256, 256, norm_act=norm_layer, pooling_size=(84, 84))
            cls = nn.Conv2d(256, num_classes, 1)

            if pretrained:
                assert os.path.exists(pretrained_path)
                data = torch.load(pretrained_path)
                body.load_state_dict(data["state_dict"]["body"])
                head.load_state_dict(data["state_dict"]["head"])
                cls.load_state_dict(data["state_dict"]["cls"])

            head = nn.Sequential(head, cls)

        elif model == "cabinet":
            body = model_dict[backbone](
                pretrained=pretrained, dilated=dilated, norm_layer=norm_layer, root=root
            )

            head = LadderHead(
                base_inchannels=[256, 512, 1024, 2048],
                base_outchannels=64,
                nclass=num_classes,
                out_channels=num_classes,
                norm_layer=norm_layer,
                se_loss=False,
                up_kwargs=None,
            )

            if pretrained:
                assert os.path.exists(pretrained_path)
                data = torch.load(pretrained_path)
                body.load_state_dict(data["state_dict"], strict=False)

        else:
            raise ("model not found")

        self.pretrained = body
        self.aux_classifier = aux_classifier  # disconnected
        self.head = head
        self.base_size = 2048
        self.crop_size = 1024
        self._up_kwargs = up_kwargs
        self.aux = False
        self.se_loss = False
        self.mean = mean
        self.std = std

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.pretrained(x)
        out = self.head(features)
        if isinstance(out, list):
            out = out[0]

        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        result = [out]
        return tuple(result)

    def evaluate(self, x, target=None):
        pred = self.forward(x)[0]

        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union


def get_deeplab(
    dataset="citys",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    norm_layer=None,
    **kwargs
):

    model = SimpleSegmentationModel(
        num_classes=datasets[dataset.lower()].NUM_CLASS,
        backbone=backbone,
        model="deeplab",
        norm_layer=norm_layer,
        pretrained_path="/input/pretrained/wide_resnet38_deeplab_vistas_modified.pth.tar",
    )

    return model


def get_cabinet_v2(
    dataset="citys",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    norm_layer=None,
    **kwargs
):

    model = SimpleSegmentationModel(
        num_classes=datasets[dataset.lower()].NUM_CLASS,
        backbone=backbone,
        model="cabinet",
        norm_layer=norm_layer,
        dilated=False,
        pretrained_path="/input/pretrained/modified_inplace_resnet101_for_sn.pth.tar",
    )

    return model


if __name__ == "__main__":
    from encoding.nn import SyncBatchNorm

    model = get_cabinet_v2(backbone="resnet101", norm_layer=SyncBatchNorm).cuda()
    input = torch.randn(1, 3, 1024, 1024).cuda()
    out = model(input)
    print(out["out"].shape)
