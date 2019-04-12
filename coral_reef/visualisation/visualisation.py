import os

import cv2
import matplotlib.pyplot as plt

from coral_reef.ml import utils as ml_utils


def plot_data(image, mask, colour_mapping):
    """
    Visualize image and mask
    :param image: image
    :param mask: mask
    :param colour_mapping: dict of type {<class_name>: <colour>}
    :return:
    """

    plt.figure(figsize=(15, 10))
    plt.imshow(image)

    cax = plt.imshow(mask, cmap="jet", alpha=0.5)
    # plt.axis("off")

    colour_ticks = [colour_mapping[c] for c in colour_mapping.keys()]

    cbar = plt.colorbar(cax, ticks=colour_ticks)
    cbar.ax.set_yticklabels(list(colour_mapping.keys()))

    plt.show()


def plot_batch_object(nn_input_batch, nn_target_batch, colour_mapping):
    """

    :param nn_input_batch:
    :param nn_target_batch:
    :param colour_mapping: dict of type {<class_name>: <colour>}

    :return:
    """

    # create numpy arrays and sort them correctly
    nn_input_batch = nn_input_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
    nn_target_batch = nn_target_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)

    for i in range(nn_input_batch.shape[0]):
        image = nn_input_batch[i]
        class_id_mask = nn_target_batch[i]
        mask = ml_utils.class_id_mask_to_colour_mask(class_id_mask, colour_mapping)
        plot_data(image, mask, colour_mapping)
