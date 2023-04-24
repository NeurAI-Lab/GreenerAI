from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .dfanet import *
from .fast_laddernet_se import *
from .fpn import *
from .tascnet import *
from .mobilenet_v3_seg_head import *
from .multi_head_net import *
from .danet import *
from ..in_place_abn.seg import get_deeplab
from .hrnet import get_hr_net
from .bisenet import get_bisenet


def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn

    models = {
        "bisenet": get_bisenet,
        "fcn": get_fcn,
        "pspnet": get_psp,
        "encnet": get_encnet,
        "dfanet": get_dfanet,
        "tascnet": get_tascnet,
        "tascnet_v2": get_tascnet_v2,
        "shelfnet": get_laddernet,
        "mobile_net_v3_head": get_mobile_net_v3_head,
        "multinet": get_multinet,
        "danet": get_danet,
        "deeplab": get_deeplab,
        "hrnet": get_hr_net,
    }
    return models[name.lower()](**kwargs)
