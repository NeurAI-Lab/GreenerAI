import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding_custom.datasets import datasets
from encoding_custom import backbones
from encoding_custom.models import FPN
from encoding_custom.models.base import BaseNet
from collections import OrderedDict
from encoding_custom.models.deeplab import ASPP_Module
from encoding_custom.models.LadderNetv66_small import BasicBlock


def get_channels_list(backbone):
    if "fpn" in backbone:
        return [256, 256, 256, 256]
    elif "res" in backbone:
        return [256, 256 * 2, 256 * 2 ** 2, 256 * 2 ** 3]
    elif "inception" in backbone:
        return [192, 384, 1024, 1536]


class DPM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.up_conv_1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )

        self.up_conv_2 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )

        self.down_1 = nn.Conv2d(
            in_channels, in_channels, stride=2, kernel_size=3, padding=1
        )
        self.down_2 = nn.Conv2d(
            in_channels, in_channels, stride=2, kernel_size=3, padding=1
        )

    def forward(self, x):
        x_up_1 = self.up_conv_1(x)
        x_up_2 = self.up_conv_1(x_up_1)
        x_down_1 = self.down_1(x_up_2)
        x_down_2 = x_down_1 + x_up_1
        x_down_2 = self.down_2(x_down_2)
        x_out = x_down_2 + x
        return x_out


class LadderBlock(nn.Module):
    def __init__(self, planes, layers, kernel=3, block=BasicBlock, norm_layer=None):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel - 1) / 2)
        self.inconv = block(planes, planes, norm_layer=norm_layer)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.down_module_list.append(
                block(planes * (2 ** i), planes * (2 ** i), norm_layer=norm_layer)
            )

        # use strided conv instead of pooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.down_conv_list.append(
                nn.Conv2d(
                    planes * 2 ** i,
                    planes * 2 ** (i + 1),
                    stride=2,
                    kernel_size=kernel,
                    padding=self.padding,
                )
            )

        # create module for bottom block
        self.bottom = block(
            planes * (2 ** (layers - 1)),
            planes * (2 ** (layers - 1)),
            norm_layer=norm_layer,
        )

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.up_conv_list.append(
                nn.Conv2d(
                    planes * 2 ** (layers - i - 1),
                    planes * 2 ** max(0, layers - i - 2),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.up_dense_list.append(
                block(
                    planes * 2 ** max(0, layers - i - 2),
                    planes * 2 ** max(0, layers - i - 2),
                    norm_layer=norm_layer,
                )
            )

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode="bilinear") + y

    def forward(self, x):
        out = self.inconv(x[-1])

        down_out = []
        # down branch
        for i in range(0, self.layers - 1):
            out = out + x[-i - 1]
            out = self.down_module_list[i](out)
            down_out.append(out)

            out = self.down_conv_list[i](out)
            out = F.relu(out)

        # bottom branch
        out = self.bottom(out)
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers - 1):
            # out = self.up_conv_list[j](out) + down_out[self.layers - j - 2]
            out = self._upsample_add(
                self.up_conv_list[j](out), down_out[self.layers - j - 2]
            )

            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class Decoder(nn.Module):
    def __init__(self, planes, layers, kernel=3, block=BasicBlock, norm_layer=None):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel
        self.padding = int((kernel - 1) / 2)
        self.inconv = block(planes, planes, norm_layer=norm_layer)
        # create module for bottom block
        self.bottom = block(
            planes * (2 ** (layers - 1)),
            planes * (2 ** (layers - 1)),
            norm_layer=norm_layer,
        )

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.up_conv_list.append(
                nn.Conv2d(
                    planes * 2 ** (layers - 1 - i),
                    planes * 2 ** max(0, layers - i - 2),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.up_dense_list.append(
                block(
                    planes * 2 ** max(0, layers - i - 2),
                    planes * 2 ** max(0, layers - i - 2),
                    norm_layer=norm_layer,
                )
            )
            # 256, 128, 64

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode="bilinear") + y

    def forward(self, x):
        # bottom branch
        out = self.bottom(x[-1])
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers - 1):
            out = self._upsample_add(self.up_conv_list[j](out), x[self.layers - j - 2])
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class Cabinet(BaseNet):
    def __init__(
        self,
        num_classes,
        backbone=None,
        aux=False,
        se_loss=True,
        lateral=False,
        norm_layer=None,
        dilated=False,
        multiscale=False,
        use_aspp=True,
        **kwargs
    ):
        super(Cabinet, self).__init__(
            num_classes,
            backbone,
            aux,
            se_loss,
            norm_layer=norm_layer,
            dilated=dilated,
            **kwargs
        )
        base_inchannels = (
            self.pretrained.base_inchannels
            if hasattr(self.pretrained, "base_inchannels")
            else get_channels_list(backbone)
        )

        self.head = LadderHead(
            base_inchannels=base_inchannels,
            base_outchannels=64,
            out_channels=num_classes,
            norm_layer=norm_layer,
            use_aspp=use_aspp,
            upkwargs=self._up_kwargs,
        )
        # base_forward

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)  # return 4 features from restnet backbone

        x = list(self.head(features))

        x[0] = F.upsample(x[0], imsize, **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.upsample(auxout, imsize, **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class LadderHead(nn.Module):
    def __init__(
        self,
        base_inchannels,
        base_outchannels,
        out_channels,
        norm_layer,
        use_aspp,
        upkwargs,
    ):
        super(LadderHead, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=base_inchannels[0],
            out_channels=base_outchannels,
            kernel_size=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=base_inchannels[1],
            out_channels=base_outchannels * 2,
            kernel_size=1,
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            in_channels=base_inchannels[2],
            out_channels=base_outchannels * 2 ** 2,
            kernel_size=1,
            bias=False,
        )
        # self.conv4 = nn.Conv2d(in_channels=base_inchannels[3], out_channels=base_outchannels * 2 ** 3,
        #                        kernel_size=1, bias=False)

        self.bn1 = norm_layer(base_outchannels)
        self.bn2 = norm_layer(base_outchannels * 2)
        self.bn3 = norm_layer(base_outchannels * 2 ** 2)
        # self.bn4 = norm_layer(base_outchannels * 2 ** 3)

        self.decoder = Decoder(planes=base_outchannels, layers=4, norm_layer=norm_layer)
        self.ladder = LadderBlock(
            planes=base_outchannels, layers=4, norm_layer=norm_layer
        )
        self.final = nn.Conv2d(base_outchannels, out_channels, 1)

        # self.aspp_module_1 = ASPP_Module(base_inchannels[0], [6, 12, 18], norm_layer, up_kwargs=upkwargs, out_channels=base_outchannels//2)
        # self.aspp_module_2 = ASPP_Module(base_inchannels[1], [6, 12, 18], norm_layer, up_kwargs=upkwargs, out_channels=base_outchannels//2 * 2)
        # self.aspp_module_3 = ASPP_Module(base_inchannels[2], [6, 12, 18], norm_layer, up_kwargs=upkwargs, out_channels=base_outchannels//2 * 2 ** 2)

        if use_aspp:
            self.top_layer = ASPP_Module(
                base_inchannels[3],
                [6, 12, 18],
                norm_layer,
                up_kwargs=upkwargs,
                out_channels=base_outchannels * 2 ** 3,
            )
        else:
            self.top_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=base_inchannels[3],
                    out_channels=base_outchannels * 2 ** 3,
                    kernel_size=1,
                    bias=False,
                ),
                norm_layer(base_outchannels * 2 ** 3),
                DPM(base_outchannels * 2 ** 3),
            )

    def forward(self, x):
        x1, x2, x3, x4 = x

        out1 = self.conv1(x1)  # 256 --> 64
        out1 = self.bn1(out1)
        out1 = F.relu(out1)

        out2 = self.conv2(x2)  # 512 --> 128
        out2 = self.bn2(out2)
        out2 = F.relu(out2)

        out3 = self.conv3(x3)  # 1024 --> 256
        out3 = self.bn3(out3)
        out3 = F.relu(out3)

        # out4 = self.conv4(x4)  # 2048 --> 512
        # out4 = self.bn4(out4)
        # out4 = F.relu(out4)

        # out1 = self.aspp_module_1(x1)
        # out2 = self.aspp_module_2(x2)
        # out3 = self.aspp_module_3(x3)
        out4 = self.top_layer(x4)

        out = self.decoder([out1, out2, out3, out4])
        out = self.ladder(out)

        pred = [self.final(out[-1])]

        # if self.se_loss:
        #     enc = F.max_pool2d(out[0], kernel_size=out[0].size()[2:])
        #     enc = torch.squeeze(enc, -1)
        #     enc = torch.squeeze(enc, -1)
        #     se = self.selayer(enc)
        #     pred.append(se)

        return pred


# todo: clean up this method
def get_cabinet_v2(
    dataset="pascal_voc",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    **kwargs
):

    model = Cabinet(
        datasets[dataset.lower()].NUM_CLASS, backbone, use_aspp=True, **kwargs
    )
    return model


def get_cabinet_v3(
    dataset="pascal_voc",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    **kwargs
):

    model = Cabinet(
        datasets[dataset.lower()].NUM_CLASS, backbone, use_aspp=False, **kwargs
    )
    return model


if __name__ == "__main__":
    A = torch.rand(2, 3, 1024, 1024).cuda()
    net = get_cabinet_v2(
        "mapillary", "resnet101", norm_layer=torch.nn.BatchNorm2d
    ).cuda()
    out = net(A)
    print(out[0].shape)
