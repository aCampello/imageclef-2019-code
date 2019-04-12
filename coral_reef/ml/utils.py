import warnings
import os
import shutil
import json

import numpy as np
import torch


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
        torch.save(model, file_path)
        if is_best:
           shutil.copyfile(file_path, os.path.join(self.folder_path, 'model_best.pth'))

    def save_instructions(self):
        with open(os.path.join(self.folder_path, "instructions.json"), "w") as fp:
            json.dump(self.instructions, fp, indent=4, sort_keys=True)
