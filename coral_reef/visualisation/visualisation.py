import os

import cv2
import matplotlib.pyplot as plt


def plot_data(image, mask, colour_mapping):
    """
    Visualize image and mask
    :param image: image
    :param mask: mask
    :param colour_mapping: dict of type {<class_name>: <colour>}
    :return:
    """

    plt.figure(figsize=(15,10))
    plt.imshow(image)

    cax = plt.imshow(mask, cmap="jet", alpha=0.5)
    plt.axis("off")

    colour_ticks = [colour_mapping[c] for c in colour_mapping.keys()]

    cbar = plt.colorbar(cax, ticks=colour_ticks)
    cbar.ax.set_yticklabels(list(colour_mapping.keys()))


    plt.show()
