import warnings
import os
import shutil
import json

import numpy as np
import torch

from tensorboardX import SummaryWriter


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


def colour_mask_to_class_id_mask(colour_mask, colour_mapping):
    """
    :param colour_mask:
    :param colour_mapping:
    :return:
    """

    class_id_mask = np.zeros(colour_mask.shape[:2]).astype(np.uint8)

    for i, k in enumerate(sorted(colour_mapping.keys())):
        class_id_mask[colour_mask == colour_mapping[k]] = i

    return class_id_mask


def class_id_mask_to_colour_mask(class_id_mask, colour_mapping):
    """
    :param class_id_mask:
    :param colour_mapping:
    :return:
    """

    colour_mask = np.zeros(class_id_mask.shape[:2]).astype(np.uint8)

    for i, k in enumerate(sorted(colour_mapping.keys())):
        colour_mask[class_id_mask == i] = colour_mapping[k]

    return colour_mask


def one_hot_to_mask(one_hot, colour_mapping):
    """
    Turn a one_hot_encoded array mask that contains colours
    :param one_hot:
    :param colour_mapping: dict of type {<class_name>: <colour>}

    :return:
    """
    mask = np.zeros(one_hot.shape[:2]).astype(np.uint8)

    for i, k in enumerate(sorted(colour_mapping.keys())):
        mask[one_hot[:, :, i] == 1] = colour_mapping[k]

    return mask


def calculate_class_weights(class_stats_file_path, colour_mapping):
    with open(class_stats_file_path, "r") as fp:
        stats = json.load(fp)

    shares = [stats[c]["share"] for c in sorted(colour_mapping.keys())]
    shares = np.array(shares)
    class_weights = 1 / np.log(1.01 + shares)
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
