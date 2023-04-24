"""
# Author  : Noah Zhang
# File    : visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

image = [
    "24--9XvrREjXs5RNbtvLfw",
    "_F2SFIm0YzTzrDuuDZP_BA",
    "GrxjXVQXSruxnFWGKKjklw",
    "uaDJRn9rShwZUny8zOwZew",
]

width, height = 43632, 30456
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(20, 15))
plt.subplots_adjust(hspace=0.1, wspace=0.2)

for i in range(len(image)):
    pre = os.path.join(
        "/volumes2/Documents/ShelfNet/experiments/segmentation/outdir",
        image[i] + ".png",
    )
    data = os.path.join(
        "/volumes2/Downloads/data sets/mapillary/validation/images", image[i] + ".jpg"
    )
    lbl = os.path.join(
        "/volumes2/Downloads/data sets/mapillary/validation/labels", image[i] + ".png"
    )

    pre = Image.open(pre)
    width += pre.size[0]
    height += pre.size[1]

    data = Image.open(data)
    width += data.size[0]
    height += data.size[1]

    lbl = Image.open(lbl)
    width += lbl.size[0]
    height += lbl.size[1]

    ax[i][0].imshow(data)
    ax[i][0].get_xaxis().set_visible(False)
    ax[i][0].get_yaxis().set_visible(False)

    ax[i][1].imshow(lbl)
    ax[i][1].get_xaxis().set_visible(False)
    ax[i][1].get_yaxis().set_visible(False)

    ax[i][2].imshow(pre)
    ax[i][2].get_xaxis().set_visible(False)
    ax[i][2].get_yaxis().set_visible(False)

    if i == 0:
        ax[i][0].set_title("Input")
        ax[i][1].set_title("Ground Truth")
        ax[i][2].set_title("Prediction")

plt.savefig("my_fig.png", dpi=400, bbox_inches="tight")
# plt.show()
