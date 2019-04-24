import json
import os
import warnings

import cv2
import numpy as np
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset

from coral_reef.constants import strings as STR
from coral_reef.ml import utils as ml_utils


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
        image = load_image(file_path_image)

        return image

    def load_nn_target(self, index):
        item = self.image_data[index]
        file_path_mask = os.path.join(self.image_base_dir, item[STR.MASK_NAME])
        mask = load_image(file_path_mask)[:, :, 0]

        class_id_mask = ml_utils.colour_mask_to_class_id_mask(mask, self.colour_mapping)

        return class_id_mask

    def __getitem__(self, index):
        image = self.load_nn_input(index)
        mask = self.load_nn_target(index)

        sample = {STR.NN_INPUT: image,
                  STR.NN_TARGET: mask}

        if self.transformation:
            sample = self.transformation(sample)

        return sample


class Normalize:

    def __init__(self):
        pass

    def __call__(self, sample):
        sample[STR.NN_INPUT] = sample[STR.NN_INPUT].astype(np.float32) / 255.0
        return sample


class RandomCrop:
    """
    Randomly crop multiple sections within the given [min_size, max_size] from an image.
    """

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


class Flip:
    """
    Flips numpy arrays randomly horizontally and vertically
    """

    def __init__(self, p_vertical=0.5, p_horizontal=0.5):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical

    def __call__(self, sample):
        nn_input = sample[STR.NN_INPUT]
        nn_target = sample[STR.NN_TARGET]

        if nn_input.ndim == 4:

            flip_vertical = np.where((np.random.rand(nn_input.shape[0]) < self.p_vertical) * 1)[0]
            nn_input[flip_vertical] = nn_input[flip_vertical, ::-1, :, :]
            nn_target[flip_vertical] = nn_target[flip_vertical, ::-1, :]

            flip_horizontal = np.where((np.random.rand(nn_input.shape[0]) < self.p_horizontal) * 1)[0]
            nn_input[flip_horizontal] = nn_input[flip_horizontal, :, ::-1, :]
            nn_target[flip_horizontal] = nn_target[flip_horizontal, :, ::-1]

        else:
            if np.random.rand() < self.p_vertical:
                nn_input = nn_input[::-1, :, :]
                nn_target = nn_target[::-1, :]

            if np.random.rand() < self.p_horizontal:
                nn_input = nn_input[:, ::-1, :]
                nn_target = nn_target[:, ::-1]

        sample[STR.NN_INPUT] = nn_input
        sample[STR.NN_TARGET] = nn_target

        return sample


class Resize:

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        nn_input = sample[STR.NN_INPUT]
        nn_target = sample.get(STR.NN_TARGET, None)

        created_list = False
        if not isinstance(nn_input, list):
            nn_input = [nn_input]
            nn_target = [nn_target] if nn_target is not None else None
            created_list = True

        out_image = np.zeros((len(nn_input), self.size, self.size, nn_input[0].shape[2])).astype(nn_input[0].dtype)

        if nn_target is not None:
            out_mask = np.zeros((len(nn_input), self.size, self.size)).astype(nn_target[0].dtype)

        for i in range(len(nn_input)):
            image = nn_input[i]
            factor = self.size / image.shape[0]

            scaled_image = zoom(image, [factor, factor, 1], order=1)
            out_image[i, :scaled_image.shape[0], :scaled_image.shape[1]] = scaled_image

            if nn_target is not None:
                scaled_mask = zoom(nn_target[i], [factor, factor], order=0)
                out_mask[i, :scaled_mask.shape[0], :scaled_mask.shape[1]] = scaled_mask

        if created_list:
            out_image = out_image[0]
            out_mask = out_mask[0] if nn_target is not None else None

        sample[STR.NN_INPUT] = out_image
        if nn_target is not None:
            sample[STR.NN_TARGET] = out_mask

        return sample


class ToTensor:
    """
    Transforms a sample to a PyTorch tensor
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        """

        :param sample:
        :return:
        """
        nn_input = sample[STR.NN_INPUT]

        nn_target = sample.get(STR.NN_TARGET, None)

        if nn_input.ndim == 3:
            nn_input = np.expand_dims(nn_input, 0)
            nn_target = np.expand_dims(nn_target, 0) if nn_target is not None else None

        ordering_input = [0, 3, 1, 2]

        sample[STR.NN_INPUT] = torch.from_numpy(nn_input.transpose(*ordering_input).copy())

        if nn_target is not None:
            sample[STR.NN_TARGET] = torch.from_numpy(nn_target.copy())

        return sample


def custom_collate(samples):
    """
    The normal collate function concatenates the tensors along a NEW first axis to create batch objects. This method can
    deal with objects that already have four dimensions - it concatenates them along the EXISTING first axis
    :param samples: List of sample objects
    :return: Sample with batch objects
    """
    out_batch_images = []
    out_batch_masks = []

    for sample in samples:
        nn_input = sample[STR.NN_INPUT]
        nn_target = sample[STR.NN_TARGET]

        # if objects only have 3 dimensions, create additional one
        if len(nn_input.shape) == 3:
            nn_input.unsqueeze(axis=0)
            nn_target.expand_dims(axis=0)

        # concatenate objects
        for i in range(nn_input.shape[0]):
            out_batch_images.append(nn_input[i])
            out_batch_masks.append(nn_target[i])

    # create batch tensor
    out_batch_images = torch.stack(out_batch_images)
    out_batch_masks = torch.stack(out_batch_masks)

    return {STR.NN_INPUT: out_batch_images, STR.NN_TARGET: out_batch_masks}
