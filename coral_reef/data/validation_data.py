import json
import os
import uuid
import glob

import cv2
from tqdm import tqdm

from coral_reef.ml.utils import cut_windows
from coral_reef.constants import paths
from coral_reef.constants import strings as STR


def create_validation_data(image_files, mask_files, output_folder_name, window_sizes=None, step_sizes=None):
    """
    Crop big images into smaller ones, which can then be used for validation. This is done, so that there is no
    randomness in the validation data (training data is randomly cropped on the fly during training).
    The crops are done in an overlapping, sliding-window manner, the relevant parameters are window_sizes and
    step_sizes. window_sizes and step_sizes must have the same length, as the elements are used together pairwise.
    :param image_files: list of paths to validation images
    :param mask_files: list of paths to validation masks
    :param output_folder_name: folder in which the cropped images will be stored
    :param window_sizes: list of integers specifying the crop sizes
    :param step_sizes: list of integers specifying the step sizes
    :return:
    """

    # create output folder
    output_folder_path = os.path.join(paths.DATA_FOLDER_PATH, output_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)

    # list that holds json output data
    data = []

    # define parameters for crops
    window_sizes = window_sizes if window_sizes is not None else [500, 1000, 1400]
    step_sizes = step_sizes if step_sizes is not None else [300, 750, 1000]

    print("Creating crops")

    # define progressbar
    pbar = tqdm(zip(image_files, mask_files), ncols=50)

    for image_file, mask_file in pbar:

        image = cv2.imread(image_file)[:, :, ::-1]
        mask = cv2.imread(mask_file)[:, :, 0]

        # cut big image into several, overlapping smaller images
        cuts, start_points = [], []
        for w_s, s_s in zip(window_sizes, step_sizes):
            cts, pts = cut_windows(image, window_size=w_s, step_size=s_s)
            cuts += cts
            start_points += pts

        # cut corresponding masks and save both cropped images and cropped masks
        for cut_image, start_point in zip(cuts, start_points):
            # cut mask
            start_x, start_y = start_point
            end_x = start_x + cut_image.shape[1]
            end_y = start_y + cut_image.shape[0]

            cut_mask = mask[start_y:end_y, start_x:end_x]

            # save mask and image
            base_name = str(uuid.uuid4())
            image_name = base_name + ".jpg"
            mask_name = base_name + "_mask.png"

            cv2.imwrite(os.path.join(output_folder_path, image_name), cut_image[:, :, ::-1])
            cv2.imwrite(os.path.join(output_folder_path, mask_name), cut_mask)

            data.append({
                STR.IMAGE_NAME: os.path.join(output_folder_name, image_name),
                STR.MASK_NAME: os.path.join(output_folder_name, mask_name)
            })

    # save data in json file
    with open(os.path.join(paths.DATA_FOLDER_PATH, "data_valid_cropped.json"), "w") as fp:
        json.dump(data, fp, indent=4)


def main():
    # get images paths that belong to uncropped validation images
    data_file_train = os.path.join(paths.DATA_FOLDER_PATH, "data_valid_BIG.json")
    with open(data_file_train, "r") as fp:
        data_valid = json.load(fp)

    image_files = [os.path.join(paths.DATA_FOLDER_PATH, d[STR.IMAGE_NAME]) for d in data_valid]
    mask_files = [os.path.join(paths.DATA_FOLDER_PATH, d[STR.MASK_NAME]) for d in data_valid]

    create_validation_data(image_files=image_files, mask_files=mask_files, output_folder_name="validation")


if __name__ == "__main__":
    main()
