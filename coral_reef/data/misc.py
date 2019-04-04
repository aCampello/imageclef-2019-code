import os
import glob
import json
from pprint import pprint

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
from coral_reef.constants import paths

def shrink_images(input_folder_path, output_folder_path, scale_factor=0.25, file_extension="png"):
    """
    Load images from input_folder_path, scale them down and save them in the output folder.
    The smaller images can then be used in other methods to calculate statistics, retrieve the name for a mask with
    certain qualities (e.g. containing a certain class) etc. more quickly
    :param input_folder_path: Folder containing the images to be scaled down.
    :param output_folder_path: Folder for the resulting images. Will be created if it does not exist
    :param scale_factor: Factor by which the images will be scaled
    :param file_extension: Extension of the input and output images
    :return: Nothing
    """
    print("Shrinking images")

    # get image files
    image_files = glob.glob(os.path.join(input_folder_path, "*." + file_extension))

    # create output folder if it does not exist
    os.makedirs(output_folder_path, exist_ok=True)

    # create process bar
    pbar = tqdm(total=len(image_files), ncols=50)

    for image_file in image_files:
        image = cv2.imread(image_file)

        # shrink image
        order = 0  # order 0 so that no new values (=classes) are introduced
        image = ndimage.zoom(image, [scale_factor, scale_factor, 1], order=order)

        # write image
        _, name = os.path.split(image_file)
        cv2.imwrite(os.path.join(output_folder_path, name), image)

        pbar.update(1)

    pbar.close()


def get_mask_names_for_color(mask_files, colour, fetch_type="all", file_extension="png"):
    """
    Returns image_paths for a given fetch_type
    :param mask_files: list containing paths to mask files
    :param colour: color corresponding to the required class
    :param fetch_type: either "all" or "max". "all" fetches all images, "max" returns image with the max pixel count of
    that class
    :param file_extension: extension of the images
    :return: a list of file_paths
    """

    assert fetch_type in ["all", "max"]

    res_paths = []

    if fetch_type == "max":
        curr_max = -1

    for mask_file in mask_files:
        # load image
        image = cv2.imread(mask_file)[:, :, 0]
        # calculate the pixel count in the image
        cnt = np.sum(image == colour)

        # remove path
        _, name = os.path.split(mask_file)

        # decide if to add/replace current image
        if fetch_type == "max" and cnt > curr_max:
            res_paths = [name]
            curr_max = cnt
        elif fetch_type == "all" and cnt > 0:
            res_paths.append(name)

    return res_paths


def calc_class_shares(mask_files, colour_mapping, file_extension="png"):
    """
    Calculates the class distribution for given mask files
    :param mask_files: list containing paths to mask files
    :param colour_mapping: dict of type {<class_name>: <colour>}
    :param file_extension: extension of the images
    :return: dict containing shares. Keys are the names of the classes
    """

    print("Calculating class shares")

    # dict containing the counts for each class
    counts = {c: 0 for c in colour_mapping.keys()}

    # create process bar
    pbar = tqdm(total=len(mask_files), ncols=50)

    for mask_file in mask_files:
        mask = cv2.imread(mask_file)[:, :, 0]

        # add up the pixel count for the individual classes
        for c in colour_mapping.keys():
            counts[c] += np.sum((mask == colour_mapping[c]) * 1)

        pbar.update(1)

    pbar.close()

    # calc relative share
    total_count = np.sum([counts[c] for c in counts.keys()])
    counts = {c: np.round(counts[c] / total_count, 5) for c in counts.keys()}

    return counts


def calculate_class_stats(mask_files, colour_mapping):
    """
    Creates a dict containing information for the different classes (names of images containing the class, name of the
    image containing the most pixel of the class, etc)
    :param mask_files: list containing paths to mask images
    :param colour_mapping: dict of type {<class_name>: <colour>}
    :return: dict containing information on the classes. Keys are the names of the classes
    """
    print("Calculating stats")

    # prepare result dict
    classes = list(colour_mapping.keys())
    stats = {c: {} for c in classes}

    # calculate class shares
    shares = calc_class_shares(mask_files, colour_mapping)

    # create process bar
    pbar = tqdm(total=len(classes), ncols=50)

    print("Retrieving filenames ")
    for c in classes:
        stats[c]["share"] = shares[c]
        stats[c]["max_file"] = get_mask_names_for_color(mask_files, colour_mapping[c], fetch_type="max")[0]
        stats[c]["files"] = get_mask_names_for_color(mask_files, colour_mapping[c], fetch_type="all")
        pbar.update(1)

    pbar.close()

    return stats


if __name__ == "__main__":
    # in_folder_path = "/home/aljo/filament/coral_reef/data/masks"
    # out_folder_path = "/home/aljo/filament/coral_reef/data/masks_super_small"
    # shrink_images(in_folder_path, out_folder_path, scale_factor=0.1)

    # mask_folder_path = "/home/aljo/filament/coral_reef/data/masks_small"
    mappings_file_path = "/home/aljo/filament/coral_reef/data/colour_mapping.json"
    with open(mappings_file_path, "r") as fp:
        colour_mapping = json.load(fp)
    #

    for d_type in ["train", "valid"]:

        with open(os.path.join(paths.DATA_FOLDER_PATH, "data_" + d_type + ".json"), "r") as fp:
            data = json.load(fp)

        mask_files = [os.path.join(paths.DATA_FOLDER_PATH, "masks_small", os.path.split(d["mask_name"])[1]) for d in data]


        stats = calculate_class_stats(mask_files, colour_mapping)

        with open(os.path.join(paths.DATA_FOLDER_PATH, "class_stats_" + d_type + ".json"), "w") as fp:
            json.dump(stats, fp, indent=4)

    # a = 1
    # pprint(calc_class_shares(mask_folder_path, colour_mapping))
