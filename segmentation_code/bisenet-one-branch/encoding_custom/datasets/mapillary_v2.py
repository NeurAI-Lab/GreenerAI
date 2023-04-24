import torch
import glob
import os
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import random
from .base import BaseDataset

CLASS = sorted(
    [
        150,
        2,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        28,
        31,
        33,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        61,
        62,
        63,
        64,
        65,
        68,
        69,
        70,
        71,
        75,
        78,
        80,
        99,
        105,
        132,
    ]
)


class MapillaryV2(BaseDataset):
    NUM_CLASS = 66

    def __init__(
        self,
        root=os.path.expanduser("~/raid/datasets/Mapillary_v1.1"),
        split="train",
        mode=None,
        transform=None,
        target_transform=None,
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
            image_path = os.path.join(root, "testing/images")
            label_path = os.path.join(root, "testing/labels")
            self.t = transforms.Compose([transforms.ToTensor()])

        else:
            print("Please check split, must be one of: train, val, test ")
        self.image_list = glob.glob(os.path.join(image_path, "*.jpg"))
        self.image_name = [x.split("/")[-1].split(".")[0] for x in self.image_list]
        self.label_list = [
            os.path.join(label_path, x + ".png") for x in self.image_name
        ]

    def _sync_transform_mapillary(self, img, label):
        img = self.t(img)

        return img, label

    def _sync_transform_mapillary2(self, img):
        img = self.t(img)
        return img

    def __getitem__(self, index):
        if self.mode == "test":
            img = Image.open(self.image_list[index]).convert("RGB")
            img = np.array(img)
            img = self._sync_transform_mapillary2(img)

            return img, os.path.basename(self.image_list[index])

        img = Image.open(self.image_list[index]).convert("RGB")
        label = Image.open(self.label_list[index])

        if self.mode == "train":
            img, label = self._sync_transform(img, label)

            img = np.array(img)
            img, label = self._sync_transform_mapillary(img, label)

        elif self.mode == "val":
            img, label = self._val_sync_transform(img, label)
            img = np.array(img)
            img, label = self._sync_transform_mapillary(img, label)

        return img, label

    def __len__(self):

        return len(self.image_list)

    def make_pred(self, mask):
        print(mask)
        # return lable images
        return CLASS[mask[0, ...]].astype(np.uint8)
