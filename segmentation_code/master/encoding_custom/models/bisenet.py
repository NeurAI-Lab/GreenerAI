"""NOTE: This BiSeNet code is integrated form TorchSeg github.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional

from encoding_custom.models.base import BaseNet
from encoding_custom.datasets import datasets

from .seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from .seg_opr.seg_oprs import ConvBnRelu, AttentionRefinement, FeatureFusion


class BiSeNet(nn.Module):
    def __init__(
        self,
        out_planes,
        backbone_channels=[64, 128, 256, 512],
        norm_layer=nn.BatchNorm2d,
    ):
        super(BiSeNet, self).__init__()

        self.business_layer = []

        self.spatial_path = SpatialPath(3, 128, norm_layer)

        conv_channel = 128
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(
                backbone_channels[-1],
                conv_channel,
                1,
                1,
                0,
                has_bn=True,
                has_relu=True,
                has_bias=False,
                norm_layer=norm_layer,
            ),
        )

        # stage = [512, 256, 128, 64]
        arms = [
            AttentionRefinement(backbone_channels[-1], conv_channel, norm_layer),
            AttentionRefinement(backbone_channels[-2], conv_channel, norm_layer),
        ]
        refines = [
            ConvBnRelu(
                conv_channel,
                conv_channel,
                3,
                1,
                1,
                has_bn=True,
                norm_layer=norm_layer,
                has_relu=True,
                has_bias=False,
            ),
            ConvBnRelu(
                conv_channel,
                conv_channel,
                3,
                1,
                1,
                has_bn=True,
                norm_layer=norm_layer,
                has_relu=True,
                has_bias=False,
            ),
        ]

        heads = [
            BiSeNetHead(conv_channel, out_planes, 16, True, norm_layer),
            BiSeNetHead(conv_channel, out_planes, 8, True, norm_layer),
            BiSeNetHead(conv_channel * 2, out_planes, 8, False, norm_layer),
        ]

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2, 1, norm_layer)

        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)

        self.business_layer.append(self.spatial_path)
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)

    def forward(self, data, context_blocks, label=None):
        spatial_out = self.spatial_path(data)

        context_blocks.reverse()

        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(
            global_context,
            size=context_blocks[0].size()[2:],
            mode="bilinear",
            align_corners=True,
        )

        last_fm = global_context
        pred_out = []

        for i, (fm, arm, refine) in enumerate(
            zip(context_blocks[:2], self.arms, self.refines)
        ):
            fm = arm(fm)
            fm += last_fm
            last_fm = F.interpolate(
                fm,
                size=(context_blocks[i + 1].size()[2:]),
                mode="bilinear",
                align_corners=True,
            )
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)
        # concate_fm = self.heads[-1](concate_fm)
        pred_out.append(concate_fm)

        #     aux_loss0 = self.criterion(self.heads[0](pred_out[0]), label)
        #     aux_loss1 = self.criterion(self.heads[1](pred_out[1]), label)
        #     main_loss = self.criterion(self.heads[-1](pred_out[2]), label)
        #     loss = main_loss + aux_loss0 + aux_loss1
        if self.training:
            return (
                self.heads[-1](pred_out[2]),
                self.heads[1](pred_out[1]),
                self.heads[0](pred_out[0]),
            )
        else:
            return (self.heads[-1](pred_out[2]),)


class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(
            in_planes,
            inner_channel,
            7,
            2,
            3,
            has_bn=True,
            norm_layer=norm_layer,
            has_relu=True,
            has_bias=False,
        )
        self.conv_3x3_1 = ConvBnRelu(
            inner_channel,
            inner_channel,
            3,
            2,
            1,
            has_bn=True,
            norm_layer=norm_layer,
            has_relu=True,
            has_bias=False,
        )
        self.conv_3x3_2 = ConvBnRelu(
            inner_channel,
            inner_channel,
            3,
            2,
            1,
            has_bn=True,
            norm_layer=norm_layer,
            has_relu=True,
            has_bias=False,
        )
        self.conv_1x1 = ConvBnRelu(
            inner_channel,
            out_planes,
            1,
            1,
            0,
            has_bn=True,
            norm_layer=norm_layer,
            has_relu=True,
            has_bias=False,
        )

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class BiSeNetHead(nn.Module):
    def __init__(
        self, in_planes, out_planes, scale, is_aux=False, norm_layer=nn.BatchNorm2d
    ):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(
                in_planes,
                256,
                3,
                1,
                1,
                has_bn=True,
                norm_layer=norm_layer,
                has_relu=True,
                has_bias=False,
            )
        else:
            self.conv_3x3 = ConvBnRelu(
                in_planes,
                64,
                3,
                1,
                1,
                has_bn=True,
                norm_layer=norm_layer,
                has_relu=True,
                has_bias=False,
            )
        if is_aux:
            self.conv_1x1 = nn.Conv2d(
                256, out_planes, kernel_size=1, stride=1, padding=0
            )
        else:
            self.conv_1x1 = nn.Conv2d(
                64, out_planes, kernel_size=1, stride=1, padding=0
            )
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(
                output, scale_factor=self.scale, mode="bilinear", align_corners=True
            )

        return output


class BiSeNet_wrapper(BaseNet):
    def __init__(self, num_classes, backbone, aux=False, se_loss=False, **kwargs):
        super().__init__(num_classes, backbone, aux, se_loss, **kwargs)
        self.head = BiSeNet(num_classes, self.pretrained.base_inchannels)
        # self.pretrained.base_inchannels[-2],
        self.aux_indexes = [1, 2]
        self.num_outputs = 3

    def forward(self, img):
        context_blocks = self.base_forward(img)
        results = self.head(img, list(context_blocks))
        return results


def get_bisenet(
    dataset="", backbone="", pretrained=False, root="~/.encoding/models", **kwargs
):
    return BiSeNet_wrapper(
        datasets[dataset.lower()].NUM_CLASS,
        backbone,
        pretrained=pretrained,
        root=root,
        **kwargs,
    )
