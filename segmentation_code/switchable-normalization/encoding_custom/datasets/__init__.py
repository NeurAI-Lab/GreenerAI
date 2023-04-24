import warnings
from torchvision.datasets import *
from encoding.datasets.base import *
from encoding.datasets.coco import COCOSegmentation
from encoding.datasets.ade20k import ADE20KSegmentation
from encoding.datasets.pascal_voc import VOCSegmentation
from encoding.datasets.pascal_aug import VOCAugSegmentation
from encoding.datasets.pcontext import ContextSegmentation
from encoding.datasets.cityscapes import CitySegmentation
from encoding.datasets.imagenet import ImageNetDataset
from encoding.datasets.minc import MINCDataset
from encoding_custom.datasets.citys_coarse import CitySegmentationCoarse
from encoding_custom.datasets.mapillary import MapillaryResearch
from encoding_custom.datasets.mapillary_old import Mapillary as MapillaryOld

from encoding.utils import EncodingDeprecationWarning

datasets = {
    "coco": COCOSegmentation,
    "ade20k": ADE20KSegmentation,
    "pascal_voc": VOCSegmentation,
    "pascal_aug": VOCAugSegmentation,
    "pcontext": ContextSegmentation,
    "citys": CitySegmentation,
    "imagenet": ImageNetDataset,
    "minc": MINCDataset,
    "cifar10": CIFAR10,
    "citys_coarse": CitySegmentationCoarse,
    "mapillary": MapillaryResearch,
    "mapillary_old": MapillaryOld,
}

acronyms = {
    "coco": "coco",
    "pascal_voc": "voc",
    "pascal_aug": "voc",
    "pcontext": "pcontext",
    "ade20k": "ade",
    "citys": "citys",
    "minc": "minc",
    "cifar10": "cifar10",
    "citys_coarse": "citys_coarse",
    "mapillary": "mapillary",
}


def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)


def _make_deprecate(meth, old_name):
    new_name = meth.__name__

    def deprecated_init(*args, **kwargs):
        warnings.warn(
            "encoding.dataset.{} is now deprecated in favor of encoding.dataset.{}.".format(
                old_name, new_name
            ),
            EncodingDeprecationWarning,
        )
        return meth(*args, **kwargs)

    deprecated_init.__doc__ = r"""
    {old_name}(...)
    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.{new_name}`.
    See :func:`~torch.nn.init.{new_name}` for details.""".format(
        old_name=old_name, new_name=new_name
    )
    deprecated_init.__name__ = old_name
    return deprecated_init


get_segmentation_dataset = _make_deprecate(get_dataset, "get_segmentation_dataset")
