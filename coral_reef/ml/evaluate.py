import os
import sys

import cv2
import torch
from tqdm import tqdm
import numpy as np

from coral_reef.constants import paths
from coral_reef.ml import predict
from coral_reef.ml.utils import colour_mask_to_class_id_mask, cut_windows

sys.path.extend([paths.DEEPLAB_FOLDER_PATH, os.path.join(paths.DEEPLAB_FOLDER_PATH, "utils")])

from metrics import Evaluator


def evaluate(image_file_paths, gt_file_paths, model, nn_input_size, num_classes, window_sizes=None, step_sizes=None,
             device=None):
    """
    Evaluate model performance
    :param image_file_paths: paths to images that will be predicted
    :param gt_file_paths: paths to ground truth masks
    :param model: model used for prediction
    :param nn_input_size: size that the images will be scaled to before feeding them into the nn
    :param num_classes: number of classes
    :param window_sizes: list of sizes that determine how the image will be cut (for sliding window). Image will be cut
    into squares
    :param step_sizes: list of step sizes for sliding window
    :param device: PyTorch device (cpu or gpu)
    :return: list of prediction masks if res_fcn is None, else Nothing
    """
    model.eval()
    evaluator = Evaluator(num_classes)

    def ev(pred_class_id_mask, index):
        gt_colour_mask = cv2.imread(gt_file_paths[index])[:, :, 0]
        gt_class_id_mask = colour_mask_to_class_id_mask(gt_colour_mask)

        # Add sample into evaluator
        evaluator.add_batch(gt_class_id_mask, pred_class_id_mask)

    with torch.no_grad():
        predict.predict(image_file_paths=image_file_paths,
                        model=model,
                        nn_input_size=nn_input_size,
                        res_fcn=ev,
                        window_sizes=window_sizes,
                        step_sizes=step_sizes,
                        device=device)

    with np.errstate(divide='ignore', invalid='ignore'):
        acc = evaluator.Pixel_Accuracy()
        acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        fWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    return acc, acc_class, mIoU, fWIoU


def find_best_cutting_sizes(image_file_paths, gt_file_paths, model, nn_input_size, num_classes, device=None):
    """
    Determine the best values for cutting the big image into smaller images for prediction
    :param image_file_paths: paths to images that will be predicted
    :param gt_file_paths: paths to ground truth masks
    :param model: model used for prediction
    :param nn_input_size: size that the images will be scaled to before feeding them into the nn
    :param num_classes: number of classes
    :param device: PyTorch device (cpu or gpu)
    :return: dict containing the tested values and the resulting prediction metrics
    """

    # define the window sizes and step sizes that will be tested
    window_sizes_list = []
    step_sizes_list = []

    step_size_fractions = [1, 1 / 2, 1 / 3]
    for ss_fraction in step_size_fractions:
        window_sizes_list.append([500, 700, 1000, 1500])
        step_sizes_list.append([int(w * ss_fraction) for w in window_sizes_list[-1]])

        window_sizes_list.append([500, 700, 1000])
        step_sizes_list.append([int(w * ss_fraction) for w in window_sizes_list[-1]])

        window_sizes_list.append([700, 1000, 1500])
        step_sizes_list.append([int(w * ss_fraction) for w in window_sizes_list[-1]])

        window_sizes_list.append([500, 700, 1000, 1500])
        step_sizes_list.append([int(w * ss_fraction) for w in window_sizes_list[-1]])

    results = []

    # go through window sizes and step sizes and use them for prediction

    for window_sizes, step_sizes in zip(window_sizes_list, step_sizes_list):
        acc, acc_class, mIoU, fWIoU = evaluate(image_file_paths=image_file_paths,
                                               gt_file_paths=gt_file_paths,
                                               model=model,
                                               nn_input_size=nn_input_size,
                                               num_classes=num_classes,
                                               window_sizes=window_sizes,
                                               step_sizes=step_sizes,
                                               device=device)

        # determine the number of images that each image was cut into
        cut_count = _determine_cut_count(image_file_paths, window_sizes, step_sizes)
        cut_count /= len(image_file_paths)

        results.append({
            "acc": np.round(acc, 2),
            "acc_class": np.round(acc_class, 2),
            "mIoU": np.round(mIoU, 2),
            "fWIoU": np.round(fWIoU, 2),
            "window_sizes": window_sizes,
            "step_sizes": step_sizes,
            "cut_count_per_image": cut_count
        })

    return results


def _determine_cut_count(image_file_paths, window_sizes, step_sizes):
    """
    Determine how many cuts the given images will be cut into based on the given parameters
    :param image_file_paths: list of images that the cuts will be calculated for
    :param window_sizes: parameter for cutting the images
    :param step_sizes: parameter for cutting the images
    :return: number of cuts
    """

    cut_count = 0
    for image_file_path in image_file_paths:
        image = cv2.imread(image_file_path)

        for window_size, step_size in zip(window_sizes, step_sizes):
            cuts, _ = cut_windows(image, window_size, step_size)
            cut_count += len(cuts)

    return cut_count
