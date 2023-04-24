import torch.nn as nn
import torch.nn.functional as F
import torch


class FPN(nn.Module):
    def __init__(self, backbone_instance, has_block_expansion=True):
        super(FPN, self).__init__()

        multiplier = 4 if has_block_expansion else 1
        # Top layer
        self.toplayer = nn.Conv2d(
            512 * multiplier, 256, kernel_size=1, stride=1, padding=0
        )  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        #        self.smooth5 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        #        self.smooth6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        #        self.smooth7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(
            256 * multiplier, 256, kernel_size=1, stride=1, padding=0
        )
        self.latlayer2 = nn.Conv2d(
            128 * multiplier, 256, kernel_size=1, stride=1, padding=0
        )
        self.latlayer3 = nn.Conv2d(
            64 * multiplier, 256, kernel_size=1, stride=1, padding=0
        )

        self.has_additional_layers = False
        self.total_outputs = 4
        self.backbone = backbone_instance

        if hasattr(self.backbone, "layer5"):
            self.has_additional_layers = True
            self.total_outputs = 7

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

    def _base_forward(self, x):
        # Bottom-up
        if hasattr(self.backbone, "layer0"):
            x = self.backbone.layer0(x)
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)
        return c1, c2, c3, c4

    def _head_forward(self, c1, c2, c3, c4):
        # Top-down
        p4 = self.toplayer(c4)
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer2(c2))
        p1 = self._upsample_add(p2, self.latlayer3(c1))
        # Smooth
        p3 = self.smooth1(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth3(p1)

        if self.has_additional_layers:
            p5 = self.backbone.layer5(c4)
            p6 = self.backbone.layer6(p5)
            p7 = self.backbone.layer7(p6)
            output = [p1, p2, p3, p4, p5, p6, p7]
        else:
            output = [p1, p2, p3, p4]

        return output

    def forward(self, x):
        # Bottom-up
        c1, c2, c3, c4 = self._base_forward(x)
        return self._head_forward(c1, c2, c3, c4)

    def forward_all(self, x):
        # Bottom-up
        c1, c2, c3, c4 = self._base_forward(x)
        return (c1, c2, c3, c4), self._head_forward(c1, c2, c3, c4)


if __name__ == "__main__":
    from encoding import backbones

    A = torch.rand(1, 3, 512, 512)
    net = FPN(
        backbones.resnet18(dilated=False, add_additional_layers=False),
        has_block_expansion=False,
    )
    out = net(A)
    for o in out:
        print(o.shape)

    net = FPN(backbones.resnet50(dilated=False, add_additional_layers=True))
    print("\n\n\n")
    out = net(A)
    for o in out:
        print(o.shape)
