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
        mask = load_image(file_path_mask)[:, :, 0]

        # catch the future warning warning
        # https://stackoverflow.com/questions/40659212/futurewarning-elementwise-codefault_collatemparison-failed-returning-scalar-but-in-the-futur
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)

            # create one-hot-encoding
            one_hot = np.zeros(mask.shape[:2] + (self.num_classes(),))

            for i, c in enumerate(self.colour_mapping.keys()):
                one_hot[mask == c, i] = 1

        return mask

    def __getitem__(self, index):
        image = self.load_nn_input(index)
        mask = self.load_nn_target(index)

        sample = {STR.NN_INPUT: image,
                  STR.NN_TARGET: mask}

        if self.transformation:
            sample = self.transformation(sample)

        return sample


class RandomCrop:

    def __init__(self, min_size, max_size, crop_count=5):
        self.min_size = min_size
        self.max_size = max_size
        self.crop_count = crop_count

    def __call__(self, sample):
        image = sample[STR.NN_INPUT]
        mask = sample[STR.NN_TARGET]
        h, w = image.shape[:2]

        nn_inputs = []
        nn_targets = []

        for i in range(self.crop_count):
            # decide crop size
            size = np.random.randint(self.min_size, self.max_size)

            # decide where to crop
            x = np.random.randint(0, w - size - 1)
            y = np.random.randint(0, h - size - 1)

            nn_inputs.append(image[y:y + size, x:x + size])
            nn_targets.append(mask[y:y + size, x:x + size])

        sample[STR.NN_INPUT] = nn_inputs
        sample[STR.NN_TARGET] = nn_targets

        return sample


class Resize:

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        nn_input = sample[STR.NN_INPUT]
        nn_target = sample[STR.NN_TARGET]

        created_list = False
        if not isinstance(nn_input, list):
            nn_input = [nn_input]
            nn_target = [nn_target]
            created_list = True

        out_image = np.zeros((len(nn_input), self.size, self.size, 3)).astype(nn_input[0].dtype)
        out_mask = np.zeros((len(nn_input), self.size, self.size)).astype(nn_target[0].dtype)

        for i, (image, mask) in enumerate(zip(nn_input, nn_target)):
            factor = self.size / image.shape[0]

            scaled_image = zoom(image, [factor, factor, 1], order=1)
            scaled_mask = zoom(mask, [factor, factor], order=0)

            out_image[i, :scaled_image.shape[0], :scaled_image.shape[1]] = scaled_image
            out_mask[i, :scaled_mask.shape[0], :scaled_mask.shape[1]] = scaled_mask

        if created_list:
            out_image = out_image[0]
            out_mask = out_mask[0]

        sample[STR.NN_INPUT] = out_image
        sample[STR.NN_TARGET] = out_mask

        return sample
