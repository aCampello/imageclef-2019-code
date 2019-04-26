import warnings
import os
import shutil
import json

import numpy as np
import torch
from coral_reef.constants import mapping

from tensorboardX import SummaryWriter


def colour_mask_to_class_id_mask(colour_mask):
    """
    :param colour_mask:
    :return:
    """
    colour_mapping = mapping.get_colour_mapping()
    class_id_mask = np.zeros(colour_mask.shape[:2]).astype(np.uint8)

    for i, k in enumerate(sorted(colour_mapping.keys())):
        class_id_mask[colour_mask == colour_mapping[k]] = i

    return class_id_mask


def class_id_mask_to_colour_mask(class_id_mask):
    """
    :param class_id_mask:
    :return:
    """
    colour_mapping = mapping.get_colour_mapping()

    colour_mask = np.zeros(class_id_mask.shape[:2]).astype(np.uint8)

    for i, k in enumerate(sorted(colour_mapping.keys())):
        colour_mask[class_id_mask == i] = colour_mapping[k]

    return colour_mask


def calculate_class_weights(class_stats_file_path, colour_mapping, modifier=1.01):
    with open(class_stats_file_path, "r") as fp:
        stats = json.load(fp)

    shares = [stats[c]["share"] for c in sorted(colour_mapping.keys())]
    shares = np.array(shares)
    class_weights = 1 / np.log(modifier + shares)
    return class_weights


def load_state_dict(model, filepath):
    pretrained_dict = torch.load(filepath, map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys and mismatching sizes
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def cut_windows(image, window_size, step_size=None):
    """
    Cut an image into several, equally sized windows. step size determines the overlap of the windows.
    :param image: image to be cut
    :param window_size: size of the square windows
    :param step_size: step size between windows, determines overlap. Depending on the image size and window size,
    it may not be possible to ensure the given step size since a constant window size is preferred
    :return: list of cut images and list of original, upper left corner points (x, y)
    """
    step_size = int(window_size / 2) if step_size is None else step_size

    h, w = image.shape[:2]
    cuts = []
    start_points = []

    for x in range(0, w, step_size):
        end_x = np.min([x + window_size, w])
        start_x = end_x - window_size

        for y in range(0, h, step_size):
            end_y = np.min([y + window_size, h])
            start_y = end_y - window_size

            pt = [start_x, start_y]
            # only add  if it hasn't been added before
            if pt not in start_points:
                cuts.append(image[start_y:end_y, start_x:end_x])
                start_points.append([start_x, start_y])

            # print("x: {}/ y:{} to x: {}/ y:{}".format(start_x, start_y, end_x, end_y))

    return cuts, start_points


def softmax(X, theta=1.0, axis=None):
    #
    # from https://nolanbconaway.github.io/blog/2017/softmax-numpy
    #
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def calc_rect_size(rect):
    """
    Calculated the area of a rectangle
    :param rect: rectangle dict
    :return: area
    """
    return (rect[3] - rect[1]) * (rect[2] - rect[0])


def calc_intersection(rect1, rect2):
    """
    Calculate the intersection area of two rectangles
    :param rect1: rectangle list
    :param rect2: rectangle list
    :return: union area
    """
    x1 = max([rect1[0], rect2[0]])
    y1 = max([rect1[1], rect2[1]])
    x2 = min([rect1[2], rect2[2]])
    y2 = min([rect1[3], rect2[3]])

    if (x2 < x1) or (y2 < y1):
        return 0

    return (x2 - x1) * (y2 - y1)


def calc_union(rect1, rect2):
    """
    Calculate the union area of two rectangles
    :param rect1: rectangle dict
    :param rect2: rectangle dict
    :return: union area
    """
    return calc_rect_size(rect1) + calc_rect_size(rect2) - calc_intersection(rect1, rect2)


def calc_IOU(rect1, rect2):
    """
    Calculate intersection over union of two rectangles. This is a measure of how similar rectangles
    :param rect1: rectangle dict
    :param rect2: rectangle dict
    :return: IOU area
    """
    return calc_intersection(rect1, rect2) / calc_union(rect1, rect2)


class Saver(object):

    def __init__(self, folder_path, instructions):
        self.instructions = instructions
        self.folder_path = folder_path

    def save_checkpoint(self, model, is_best, epoch):
        file_path = os.path.join(self.folder_path, "checkpoint_epoch_{}.pt".format(epoch))
        torch.save(model.state_dict(), file_path)
        if is_best:
            shutil.copyfile(file_path, os.path.join(self.folder_path, 'model_best.pth'))

    def save_instructions(self):
        with open(os.path.join(self.folder_path, "instructions.json"), "w") as fp:
            json.dump(self.instructions, fp, indent=4, sort_keys=True)


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer
