import os
import glob
import json
from time import time
import uuid

import cv2
import numpy as np

from coral_reef.constants import strings as STR
from coral_reef.constants import paths
from coral_reef.ml.utils import cut_windows, calc_IOU

import matplotlib.pyplot as plt


def _get_bad_prediction_rects(gt, pred, rect_size, density_thresh):
    """

    :param gt: ground truth; class_id array
    :param pred: predictions of the neural network; class_id array
    :return:
    """

    wrong_pred = (gt != pred) * 1

    stepsize = int(rect_size / 20)

    start_time = time()

    windows, pts = cut_windows(wrong_pred, window_size=rect_size, step_size=stepsize)

    densities = []

    # calculate the density for each window
    for window in windows:
        densities.append(np.sum(window) / np.prod(window.shape[:2]))

    duration = time() - start_time
    print("calculated {} densities in {}. {} per second".format(len(densities), duration, len(densities) / duration))

    del windows

    indices = np.argsort(densities)[::-1]

    out_rects = []

    IOU_thresh = 0.1
    for index in indices:
        # check if density is large enough
        if densities[index] < density_thresh:
            break
        new_rect = [*pts[index], pts[index][0] + rect_size, pts[index][1] + rect_size]

        # check if the new rect would overlap too much with an old one (which has a higher density)
        much_overlap = False
        for rect in out_rects:
            if calc_IOU(rect, new_rect) > IOU_thresh:
                much_overlap = True
                break

        # if too much overlap go one with the next rect
        if much_overlap:
            continue

        out_rects.append(new_rect)

    return out_rects


def plot(image, rects):
    plt.figure(figsize=(10, 8))

    image = np.array(image)
    image = (np.dstack((image, image, image)) * 255).astype(np.uint8)
    stroke_width = 4

    for rect in rects:
        image = cv2.rectangle(image.copy(), (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0),
                              stroke_width + 8)
        image = cv2.rectangle(image.copy(), (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0),
                              stroke_width)
    plt.imshow(image)

    plt.show()


def generate_hard_data(image_files, pred_files, gt_files, out_folder_name):
    out_folder_path = os.path.join(paths.DATA_FOLDER_PATH, out_folder_name)
    os.makedirs(out_folder_path, exist_ok=True)

    data = []

    for i, (image_file, pred_file, gt_file) in enumerate(zip(image_files, pred_files, gt_files)):

        print("generating {} of {}".format(i + 1, len(image_files)))

        image = cv2.imread(image_file)
        gt = cv2.imread(gt_file)[:, :, 0]
        pred = cv2.imread(pred_file)[:, :, 0]

        rects = _get_bad_prediction_rects(gt=gt, pred=pred, rect_size=400, density_thresh=0.5)
        rects += _get_bad_prediction_rects(gt=gt, pred=pred, rect_size=500, density_thresh=0.4)
        rects += _get_bad_prediction_rects(gt=gt, pred=pred, rect_size=800, density_thresh=0.2)
        rects += _get_bad_prediction_rects(gt=gt, pred=pred, rect_size=1000, density_thresh=0.15)

        for rect in rects:
            base_name = str(uuid.uuid4())
            out_image_name = base_name + ".jpg"
            out_mask_name = base_name + "_mask.png"

            out_image = image[rect[1]: rect[3], rect[0]:rect[2], ::-1]
            out_mask = gt[rect[1]: rect[3], rect[0]:rect[2]]

            cv2.imwrite(os.path.join(out_folder_path, out_image_name), out_image)
            cv2.imwrite(os.path.join(out_folder_path, out_mask_name), out_mask)

            data.append({
                STR.IMAGE_NAME: os.path.join(out_folder_name, out_image_name),
                STR.MASK_NAME: os.path.join(out_folder_name, out_mask_name)
            })

    with open(os.path.join(paths.DATA_FOLDER_PATH, "data_train_hard.json"), "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    predictions_folder = os.path.join(paths.DATA_FOLDER_PATH, "predictions")
    pred_files = glob.glob(os.path.join(predictions_folder, "*.png"))
    pred_files = pred_files[3:]

    gt_files = [os.path.join(paths.MASK_FOLDER_PATH, os.path.split(p)[1][:-4] + "_mask.png") for p in pred_files]
    image_files = [os.path.join(paths.IMAGE_FOLDER_PATH, os.path.split(p)[1][:-4] + ".JPG") for p in pred_files]

    generate_hard_data(image_files, pred_files, gt_files, out_folder_name="hard_crops")
