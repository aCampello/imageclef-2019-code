import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import AxesGrid

from coral_reef.ml import utils as ml_utils


def plot_data(image, gt_mask, colour_mapping, prediction_mask=None, out_file_path=None):
    """
    Visualize image and mask
    :param image: image
    :param gt_mask: mask
    :param colour_mapping: dict of type {<class_name>: <colour>}
    :param out_file_path: file to output image if the image should be saved
    :return:
    """

    class_name_converter = {
        "background": "Background",
        "c_algae_macro_or_leaves": "Algae - Macro or Leaves",
        "c_fire_coral_millepora": "Fire Coral - Millepora",
        "c_hard_coral_boulder": "Hard Coral - Boulder",
        "c_hard_coral_branching": "Hard Coral - Branching",
        "c_hard_coral_encrusting": "Hard Coral - Encrusting",
        "c_hard_coral_foliose": "Hard Coral - Foliose",
        "c_hard_coral_mushroom": "Hard Coral - Mushroom",
        "c_hard_coral_submassive": "Hard Coral - Submassive",
        "c_hard_coral_table": "Hard Coral - Table",
        "c_soft_coral": "Soft Coral",
        "c_soft_coral_gorgonian": "Soft Coral - Gorgonian",
        "c_sponge": "Sponge",
        "c_sponge_barrel": "Sponge - Barrel"
    }


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
        im = grid[i].imshow(overlay, cmap=cm.get_cmap("tab20", 14), alpha=0.8, vmin=0, vmax=255)

        colours = list(sorted(set(list(colours) + np.unique(overlays).tolist())))
        # plt.axis("off")
    grid[-1].imshow(image)
    grid[-1].set_axis_off()

    # cbar = grid[0].cax.colorbar(ax)
    cbar = grid.cbar_axes[0].colorbar(im)

    # cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
    # cbar.ax.set_yticklabels(['low', 'medium', 'high'])
    # plt.show()

    y_tick_labels = [class_name_converter[c] for c in colour_mapping.keys()]
    # cbar = fig.colorbar(fig, ticks=colour_ticks)
    cbar.ax.set_yticks(colours)

    cbar.ax.set_yticklabels(y_tick_labels)

    if out_file_path:
        plt.savefig(out_file_path)
    else:
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
        mask = ml_utils.class_id_mask_to_colour_mask(class_id_mask)
        plot_data(image, mask, colour_mapping)
