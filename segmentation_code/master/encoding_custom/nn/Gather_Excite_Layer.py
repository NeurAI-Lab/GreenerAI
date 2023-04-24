import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
import math


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


class DepthwiseConv2D(nn.Module):
    def __init__(
        self, channels, kernel_size=3, depth_multiplier=1, stride=1, padding=0
    ):
        super(DepthwiseConv2D, self).__init__()
        self.dwconv = nn.Conv2d(
            channels,
            channels * depth_multiplier,
            groups=channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        return self.dwconv(x)


class GEBlock(nn.Module):
    """
    Usage: For global gather-excites you have to provide the spatial size as the entire image/feature size!
    For example, if globalpool is True for a feature 'x' of size [b, c, h, w],
    you should use ge_theta = GEBlock(c, [h, w], 1, learned=True, globalpool=True)
    Note: Extent is supported only uptil 8. Among options [2, 4, 8]
    """

    def __init__(
        self, in_planes, spatial, stride=1, extent=1, learned=True, globalpool=True
    ):
        super(GEBlock, self).__init__()
        self.learned = learned
        self.globalpool = globalpool
        self.extent = extent
        if learned:
            self.dwconv = DepthwiseConv2D(in_planes, spatial, stride=stride, padding=0)
            self.sigmoid = nn.Sigmoid()
            self.global_GE_theta = nn.Sequential(
                self.dwconv, nn.BatchNorm2d(in_planes), self.sigmoid
            )
            if extent != 1 and not globalpool:
                assert is_power2(extent), "Extent must be 2^n, e.g. 2, 4, 8, 16 ...."
                self.dwbnrelu = nn.Sequential(
                    DepthwiseConv2D(
                        in_planes, spatial, stride=2, padding=(spatial // 2)
                    ),
                    nn.BatchNorm2d(in_planes),
                    nn.ReLU(),
                )
                self.dwbn = nn.Sequential(
                    DepthwiseConv2D(
                        in_planes, spatial, stride=2, padding=(spatial // 2)
                    ),
                    nn.BatchNorm2d(in_planes),
                )

    def forward(self, in_tensor):
        if self.learned and self.globalpool:
            # print("GE_theta_global")
            x = self.global_GE_theta(in_tensor)
            x = x * in_tensor

        elif self.learned and self.extent and not self.globalpool:
            assert self.extent > 1 and is_power2(
                self.extent
            ), "extent must be greater than 1"
            # print("GE_theta with extent {}".format(self.extent))
            x = self.dwbnrelu(in_tensor)
            down_steps = int(math.log2(self.extent))
            for n in range(down_steps - 1):
                if n >= down_steps - 2:
                    x = self.dwbn(x)
                else:
                    x = self.dwbnrelu(x)
            x = nn.UpsamplingBilinear2d(scale_factor=self.extent)(x)
            x = nn.Sigmoid()(x)
            x = x * in_tensor

        elif not self.learned and self.globalpool:
            # print("GE_theta_minus global")
            x = nn.AvgPool2d(in_tensor.shape[2:])(in_tensor)
            x = nn.Sigmoid()(x)
            x = x * in_tensor

        elif not self.learned and not self.globalpool:
            assert self.extent > 1 and is_power2(
                self.extent
            ), "extent must be greater than 1"
            # print("GE_theta_minus with extent {}".format(self.extent))
            x = nn.AvgPool2d(self.extent)(in_tensor)
            x = nn.UpsamplingBilinear2d(scale_factor=self.extent)(x)
            x = nn.Sigmoid()(x)
            x = x * in_tensor

        return x + in_tensor
