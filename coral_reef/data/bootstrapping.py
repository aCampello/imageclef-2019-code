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
    Calculate which parts of the images were badly predicted and return a list of rects that captures those areas
    :param gt: ground truth; class_id array
    :param pred: predictions of the neural network; class_id array
    :param rect_size: The size of the output rects. Specifies one side, rects are square
    :param density_thresh: what share of the rectangle should be falsely predicted so that it counts as "bad"
    :return: list of rectangles
    """

    # create a wrong prediction mask
    wrong_pred = (gt != pred) * 1

    # step size that is used (in conjunction with the rect_size) to calculate candidate crops
    stepsize = int(rect_size / 20)

    start_time = time()
    # crop windows in a sliding window manner
    windows, pts = cut_windows(wrong_pred, window_size=rect_size, step_size=stepsize)

    densities = []

    # calculate the density for each window
    for window in windows:
        densities.append(np.sum(window) / np.prod(window.shape[:2]))

    duration = time() - start_time
    # print("calculated {} densities in {}. {} per second".format(len(densities), duration, len(densities) / duration))

    del windows

    # sort the densities in descending order
    indices = np.argsort(densities)[::-1]

    out_rects = []

    # define intersection-over-union threshold that is used to discard rectangle candidates that overlap too much
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


def _plot(wrong_prediction_mask, rects):
    """
    Helper function that visualizes the outputs of the _get_bad_prediction_rects method
    :param wrong_prediction_mask: mask containing only 1s and 0s
    :param rects: list of [x1, y1, x2, y2] to be drawn on the image
    :return:
    """
    # define figure
    plt.figure(figsize=(10, 8))

    # create plottable image from wrong prediction mask
    wrong_prediction_mask = np.array(wrong_prediction_mask)
    image = (np.dstack((wrong_prediction_mask, wrong_prediction_mask, wrong_prediction_mask)) * 255).astype(np.uint8)
    stroke_width = 4

    # go through rects and paint them on the image
    for rect in rects:
        image = cv2.rectangle(image.copy(), (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0),
                              stroke_width + 8)
        image = cv2.rectangle(image.copy(), (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0),
                              stroke_width)
    # show image
    plt.imshow(image)

    plt.show()


def generate_hard_data(image_files, pred_files, gt_files, out_folder_name):
    """
    Creates image crops that contain data that the model was not able to predict correctly
    :param image_files: list of paths to images
    :param pred_files: list of paths to predicted images
    :param gt_files: list of paths to ground truth images
    :param out_folder_name: folder that will be used for the newly created data
    :return: Nothing
    """

    # create output folder
    out_folder_path = os.path.join(paths.DATA_FOLDER_PATH, out_folder_name)
    os.makedirs(out_folder_path, exist_ok=True)

    # list to store the json output data
    data = []

    # go though each prediction and create data
    for i, (image_file, pred_file, gt_file) in enumerate(zip(image_files, pred_files, gt_files)):

        print("generating {} of {}".format(i + 1, len(image_files)))

        # read image, gt and prediction
        image = cv2.imread(image_file)
        gt = cv2.imread(gt_file)[:, :, 0]
        pred = cv2.imread(pred_file)[:, :, 0]

        # calculate rects that were badly predicted
        rects = _get_bad_prediction_rects(gt=gt, pred=pred, rect_size=400, density_thresh=0.5)
        rects += _get_bad_prediction_rects(gt=gt, pred=pred, rect_size=500, density_thresh=0.4)
        rects += _get_bad_prediction_rects(gt=gt, pred=pred, rect_size=800, density_thresh=0.2)
        rects += _get_bad_prediction_rects(gt=gt, pred=pred, rect_size=1000, density_thresh=0.15)

        # go through the rectangles, crop the data accordingly and store it
        for rect in rects:

            # define output name for crop
            base_name = str(uuid.uuid4())
            out_image_name = base_name + ".jpg"
            out_mask_name = base_name + "_mask.png"
            # crop image and mask
            out_image = image[rect[1]: rect[3], rect[0]:rect[2], ::-1]
            out_mask = gt[rect[1]: rect[3], rect[0]:rect[2]]
            # save it
            cv2.imwrite(os.path.join(out_folder_path, out_image_name), out_image)
            cv2.imwrite(os.path.join(out_folder_path, out_mask_name), out_mask)
            # save information
            data.append({
                STR.IMAGE_NAME: os.path.join(out_folder_name, out_image_name),
                STR.MASK_NAME: os.path.join(out_folder_name, out_mask_name)
            })

    # save data file
    with open(os.path.join(paths.DATA_FOLDER_PATH, "data_train_hard.json"), "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    predictions_folder = os.path.join(paths.DATA_FOLDER_PATH, "predictions")
    pred_files = glob.glob(os.path.join(predictions_folder, "*.png"))
    pred_files = pred_files[3:]

    gt_files = [os.path.join(paths.MASK_FOLDER_PATH, os.path.split(p)[1][:-4] + "_mask.png") for p in pred_files]
    image_files = [os.path.join(paths.IMAGE_FOLDER_PATH, os.path.split(p)[1][:-4] + ".JPG") for p in pred_files]

    generate_hard_data(image_files, pred_files, gt_files, out_folder_name="hard_crops")
