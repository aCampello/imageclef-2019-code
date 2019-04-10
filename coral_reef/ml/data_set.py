import json
import os
import warnings

import cv2
import numpy as np
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset

from coral_reef.constants import strings as STR


def load_image(filepath):
    return cv2.imread(filepath)[:, :, ::-1]


class DictArrayDataSet(Dataset):

    def __init__(self, image_base_dir, data, colour_mapping, transformation=None):
        self.image_base_dir = image_base_dir
        self.image_data = data
        self.colour_mapping = colour_mapping
        self.transformation = transformation

    def __len__(self):
        return len(self.image_data)

    def num_classes(self):
        return len(self.colour_mapping.keys())

    def load_nn_input(self, index):
        item = self.image_data[index]
        file_path_image = os.path.join(self.image_base_dir, item[STR.IMAGE_NAME])
        image = load_image(file_path_image).astype(np.float32) / 255.0

        return image

    def load_nn_target(self, index):
        item = self.image_data[index]
        file_path_mask = os.path.join(self.image_base_dir, item[STR.MASK_NAME])
        mask = load_image(file_path_mask)

        # create one-hot-encoding
        one_hot = np.zeros(mask.shape[:2] + (self.num_classes(),))

        for i, c in enumerate(self.colour_mapping.keys()):
            one_hot[mask == c, i] = 1

        return one_hot

    def __getitem__(self, index):
        image = self.load_nn_input(index)
        mask = self.load_nn_target(index)

        sample = {STR.NN_INPUT: image,
                  STR.NN_TARGET: mask}

        if self.transformation:
            sample = self.transformation(sample)

        return sample
