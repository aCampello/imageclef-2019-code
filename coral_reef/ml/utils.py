import warnings

import numpy as np


def mask_to_one_hot(mask, colour_mapping):
    """
    Turn a mask that contains colour into a one-hot-encoded array
    :param mask:
    :param colour_mapping: dict of type {<class_name>: <colour>}

    :return:
    """

    one_hot = np.zeros(mask.shape[:2] + (len(colour_mapping.keys()),))

    for i, k in enumerate(sorted(colour_mapping.keys())):
        one_hot[mask == colour_mapping[k], i] = 1

    return one_hot


def one_hot_to_mask(one_hot, colour_mapping):
    """
    Turn a one_hot_encoded array mask that contains colours
    :param one_hot:
    :param colour_mapping: dict of type {<class_name>: <colour>}

    :return:
    """
    mask = np.zeros(one_hot.shape[:2]).astype(np.uint8)

    for i, k in enumerate(colour_mapping.keys()):
        mask[one_hot[:, :, i] == 1] = colour_mapping[k]

    return mask
