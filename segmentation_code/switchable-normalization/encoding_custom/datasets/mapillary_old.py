# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CabiNet
# Created by: Ning Zhang
#

import torch
import glob
import os
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import random
from encoding.datasets.base import BaseDataset


class _sync_transform(transforms.RandomResizedCrop):
    def _mask_transform(self, mask):
        mask = np.array(mask).astype("uint8")
        return torch.from_numpy(mask).long()

    def __call__(self, img, label):
        assert img.size == label.size

        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        return [
            F.resized_crop(img, i, j, h, w, self.size, interpolation=Image.BILINEAR),
            self._mask_transform(
                F.resized_crop(
                    label, i, j, h, w, self.size, interpolation=Image.NEAREST
                )
            ),
        ]


class Mapillary(BaseDataset):
    NUM_CLASS = 66

    def __init__(
        self,
        root=os.path.expanduser("/volumes2"),
        split="train",
        mode=None,
        transform=None,
        target_transform=None,
        test_folder="/volumes2",
        **kwargs
    ):
        super().__init__(root, split, mode, transform, target_transform, **kwargs)

        self.split = split

        if self.split == "train":
            image_path = os.path.join(root, "training/images")
            label_path = os.path.join(root, "training/labels")
            self.t = transforms.Compose([transforms.ToTensor()])

        elif self.split == "val":
            image_path = os.path.join(root, "validation/images")
            label_path = os.path.join(root, "validation/labels")
            self.t = transforms.Compose([transforms.ToTensor()])

        elif self.split == "test":
            image_path = test_folder
            label_path = os.path.join(root, "testing/labels")
            self.t = transforms.Compose([transforms.ToTensor()])

        else:
            print("Please check split, must be one of: train, val, test ")

        self.image_list = glob.glob(os.path.join(image_path, "*.jpg"))
        self.image_name = [
            os.path.splitext(os.path.basename(x))[0] for x in self.image_list
        ]
        self.label_list = [
            os.path.join(label_path, x + ".png") for x in self.image_name
        ]

    def _sync_transform_mapillary(self, img, label=None):
        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        img = self.t(img)
        random.seed(seed)

        return img, label

    def __getitem__(self, index):
        if self.mode == "test":
            img, _ = Image.open(self.image_list[index]).convert("RGB")
            img = np.array(img)
            img = self._sync_transform_mapillary(img)

            return img, os.path.basename(self.image_list[index])

        img = Image.open(self.image_list[index]).convert("RGB")
        label = Image.open(self.label_list[index])

        if self.mode == "train":
            img, label = _sync_transform(
                self.crop_size, scale=(0.5, 1.0), ratio=(5.0 / 6.0, 6.0 / 5.0)
            )(img, label)
        elif self.mode == "val":
            img, label = self._val_sync_transform(img, label)

        img = np.array(img)
        img, label = self._sync_transform_mapillary(img, label)

        return img, label

    def __len__(self):

        return len(self.image_list)
