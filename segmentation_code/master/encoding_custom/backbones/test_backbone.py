from encoding_custom import backbones
from encoding.nn.syncbn import SyncBatchNorm
import torch

# backbone = backbones.resnet50(pretrained=True, dilated=False,
#                                 norm_layer=SyncBatchNorm,
#                                 dilate_only_last_layer=False).cuda()

backbone = backbones.efficientnet_pytorch.EfficientNet.from_pretrained(
    "efficientnet-b7"
).cuda()
backbone.eval()
A = torch.rand(1, 3, 256, 256).cuda()
out = backbone(A)
for o in out:
    print(o.shape)
