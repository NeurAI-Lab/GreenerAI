import os
import sys
import random
import numpy as np
from tqdm import tqdm, trange
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform

from encoding.datasets.base import BaseDataset
import glob
import json


class MapillaryResearch(BaseDataset):
    NUM_CLASS = 65
    IGNORE_INDEX = -1

    def __init__(
        self,
        root=os.path.expanduser("~/.encoding/data"),
        split="train",
        mode=None,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        super(MapillaryResearch, self).__init__(
            root, split, mode, transform, target_transform, **kwargs
        )

        if self.split == "train":
            image_path = os.path.join(root, "training/images")
            label_path = os.path.join(root, "training/labels")

        elif self.split == "val":
            image_path = os.path.join(root, "validation/images")
            label_path = os.path.join(root, "validation/labels")

        elif self.split == "test":
            image_path = os.path.join(root, "testing/images")
            label_path = os.path.join(root, "testing/labels")

        else:
            raise ("Please check split, must be one of: train, val, test ")

        self.images = glob.glob(os.path.join(image_path, "*.jpg"))
        self.image_names = [x.split("/")[-1].split(".")[0] for x in self.images]
        self.mask_paths = [
            os.path.join(label_path, x + ".png") for x in self.image_names
        ]

        with open(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        ) as config_file:
            config = json.load(config_file)
        # in this example we are only interested in the labels
        self.labels = config["labels"]

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        if self.mode == "test":
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        # mask = self.masks[index]
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == "train":
            img, mask = self._sync_transform(img, mask)
        elif self.mode == "val":
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == "testval"
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.images)

    def make_pred(self, mask):  # needs to implmented for visulaizations
        values = np.unique(mask)
        for i in range(len(values)):
            assert values[i] in self._indices
        index = np.digitize(mask.ravel(), self._indices, right=True)
        return self._classes[index].reshape(mask.shape)

    def _mask_transform(self, mask):
        mask = np.array(mask).astype("int32")
        mask[mask == 65] = -1
        return torch.from_numpy(mask).long()

    def apply_color_map(self, image_array):
        color_array = np.zeros(
            (image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8
        )

        for label_id, label in enumerate(self.labels):
            # set all pixels with the current label to the color of the current label
            color_array[image_array == label_id] = label["color"]

        color_array = Image.fromarray(color_array)
        return color_array
