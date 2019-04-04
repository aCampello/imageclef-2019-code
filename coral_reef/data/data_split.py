import os
import glob
import json

from copy import copy

import cv2
import numpy as np
from tqdm import tqdm

from random import shuffle
from scipy import spatial
import matplotlib.pyplot as plt
from coral_reef.constants import paths


def _calc_mask_stats(mask_folder_path, colour_mapping):
    """
    create a dict containing the class distribution for each mask file
    :param mask_folder_path: Folder containing the masks
    :return: dict {<mask_name>: {}} containing class distributions for each image
    """

    print("Calculating mask stats")
    mask_stats = {}
    mask_paths = glob.glob(os.path.join(mask_folder_path, "*.png"))

    # create progressbar
    pbar = tqdm(total=len(mask_paths), ncols=50)

    for mask_path in mask_paths:
        _, name = os.path.split(mask_path)
        mask_stats[name] = {}

        mask = cv2.imread(mask_path)[:, :, 0]

        # get pixels for each class
        for c in colour_mapping.keys():
            mask_stats[name][c] = np.sum((mask == colour_mapping[c]) * 1)

        pbar.update(1)

    pbar.close()

    return mask_stats


def calc_distance_metric(weights):
    def metric(v1, v2):
        return spatial.distance.cosine(v1 * weights, v2 * weights)

    return metric


def calculate_split(mask_folder_path, colour_mapping, training_size=0.85):
    """
    Calculate a good training/validation split for the given data
    :param mask_folder_path:
    :param colour_mapping:
    :param training_size: share of the data that goes into the training set
    :return: two lists, containing the files for training and validation respectively
    """

    mask_stats = _calc_mask_stats(mask_folder_path, colour_mapping)
    names = list(sorted(mask_stats.keys()))
    classes = list(sorted(colour_mapping.keys()))

    # create a matrix containing the counts
    counts = np.zeros((len(names), len(classes)))
    for i, n in enumerate(names):
        for j, c in enumerate(classes):
            counts[i, j] = mask_stats[n][c]

    # get the distance metric function
    weights = 1 / counts.sum(axis=0) / counts.sum()
    distance_metric = calc_distance_metric(weights)

    # index where to split the data into training and validation set
    split_index = int(len(names) * training_size)

    # get initial split by doing some random splits and comparing them
    print("Calculating initial split")
    permutation_count = int(1e5)

    min_dist = 1e5
    min_distances = []
    best_permutation = []
    for i in range(permutation_count):

        permutation = np.random.permutation(len(names))
        distrib_train = counts[permutation[:split_index]].sum(axis=0)
        distrib_valid = counts[permutation[split_index:]].sum(axis=0)

        distance = distance_metric(distrib_train, distrib_valid)

        if distance < min_dist:
            print("iteration {}, distance: {:.3e}".format(i + 1, distance))
            min_dist = distance
            best_permutation = permutation
            min_distances.append(min_dist)

    indices_train = best_permutation[:split_index].tolist()
    indices_val = best_permutation[split_index:].tolist()

    print("Swapping")
    # refine splits by swapping random entries if it decreases the distance
    swap_count = int(1e4)
    for i in range(swap_count):

        idx1 = indices_train[np.random.randint(len(indices_train))]
        idx2 = indices_val[np.random.randint(len(indices_val))]

        temp_indices_train = [idx for idx in indices_train if idx != idx1] + [idx2]
        temp_indices_val = [idx for idx in indices_val if idx != idx2] + [idx1]

        distrib_train = counts[temp_indices_train].sum(axis=0)
        distrib_valid = counts[temp_indices_val].sum(axis=0)

        distance = distance_metric(distrib_train, distrib_valid)

        if distance < min_dist:
            print("iteration {}, distance: {:.3e}".format(i + 1, distance))

            min_distances.append(distance)
            min_dist = distance
            indices_train = temp_indices_train
            indices_val = temp_indices_val

    swap_count = 100
    for i in range(swap_count):
        print("{} of {} opt swaps".format(i + 1, swap_count))
        best_idx_train = -1
        best_idx_val = -1
        temp_min_dist = min_dist

        for idx1 in indices_train:
            for idx2 in indices_val:

                temp_indices_train = [idx for idx in indices_train if idx != idx1] + [idx2]
                temp_indices_val = [idx for idx in indices_val if idx != idx2] + [idx1]

                distrib_train = counts[temp_indices_train].sum(axis=0)
                distrib_valid = counts[temp_indices_val].sum(axis=0)

                distance = distance_metric(distrib_train, distrib_valid)

                if distance < temp_min_dist:
                    temp_min_dist = distance
                    best_idx_train = idx1
                    best_idx_val = idx2

        if best_idx_val != -1:
            # swap indices
            indices_train = [idx for idx in indices_train if idx != best_idx_train] + [best_idx_val]
            indices_val = [idx for idx in indices_val if idx != best_idx_val] + [best_idx_train]

            min_distances.append(temp_min_dist)
            min_dist = temp_min_dist
            print("new min distance: {:.3e}".format(min_dist))

        else:
            break

    files_train = [names[idx] for idx in indices_train]
    files_valid = [names[idx] for idx in indices_val]

    return files_train, files_valid


if __name__ == "__main__":
    mask_folder_path = "/home/aljo/filament/coral_reef/data/masks_super_small"

    mappings_file_path = "/home/aljo/filament/coral_reef/data/colour_mapping.json"
    with open(mappings_file_path, "r") as fp:
        colour_mapping = json.load(fp)

    files_train, files_valid = calculate_split(mask_folder_path, colour_mapping)

    data_train = [{"image_name": os.path.join("images", name[:-9] + ".jpg"),
                   "mask_name": os.path.join("masks", name)}
                  for name in files_train]

    data_val = [{"image_name": os.path.join("images", name[:-9] + ".jpg"),
                 "mask_name": os.path.join("masks", name)}
                for name in files_valid]

    data = data_train + data_val

    with open(os.path.join(paths.DATA_FOLDER_PATH, "data.json"), "w") as fp:
        json.dump(data, fp, indent=4)

    with open(os.path.join(paths.DATA_FOLDER_PATH, "data_train.json"), "w") as fp:
        json.dump(data_train, fp, indent=4)

    with open(os.path.join(paths.DATA_FOLDER_PATH, "data_valid.json"), "w") as fp:
        json.dump(data_val, fp, indent=4)
