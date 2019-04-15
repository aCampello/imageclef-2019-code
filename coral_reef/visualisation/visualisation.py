import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from coral_reef.ml import utils as ml_utils


def plot_data(image, gt_mask, colour_mapping, prediction_mask=None):
    """
    Visualize image and mask
    :param image: image
    :param gt_mask: mask
    :param colour_mapping: dict of type {<class_name>: <colour>}
    :return:
    """

    if gt_mask.ndim == 3:
        gt_mask = gt_mask[:, :, 0]

    fig = plt.figure(figsize=(15, 10))

    overlays = [gt_mask, prediction_mask] if prediction_mask is not None else [gt_mask]

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, len(overlays) + 1),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='left',
                    cbar_pad=0.05
                    )
    colours = set()
    for i, overlay in enumerate(overlays):
        grid[i].imshow(image)
        grid[i].set_axis_off()
        ax = grid[i].imshow(overlay, cmap="tab20", alpha=0.9)

        colours = list(sorted(set(list(colours) + np.unique(overlays).tolist())))
        # plt.axis("off")
    grid[-1].imshow(image)
    grid[-1].set_axis_off()

    # cbar = grid[-1].cax.colorbar(ax)
    cbar = grid.cbar_axes[0].colorbar(ax)

    # cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
    # cbar.ax.set_yticklabels(['low', 'medium', 'high'])
    # plt.show()

    colour_ticks = [colour_mapping[c] for c in colour_mapping.keys()]
    # cbar = fig.colorbar(fig, ticks=colour_ticks)
    cbar.ax.set_yticks(colours)

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
