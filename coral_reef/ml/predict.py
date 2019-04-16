import json
import os
import sys

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import zoom

from coral_reef.ml.data_set import Normalize, Resize, ToTensor
from coral_reef.constants import paths
from coral_reef.constants import strings as STR

from coral_reef.ml.utils import load_state_dict
from coral_reef.utils.print_utils import Printer

sys.path.extend([paths.DEEPLAB_FOLDER_PATH, os.path.join(paths.DEEPLAB_FOLDER_PATH, "utils")])
from modeling.deeplab import DeepLab


def predict_by_cutting(image, model, device, nn_input_size, window_sizes, step_sizes, verbose=0):
    """
    Predicts the given image with the given model. Image is cut into several overlapping pieces which will be predicted
    individually. Will be recombined after prediction.
    :param image: image as numpy array as uint8
    :param model: model used for prediction
    :param device: PyTorch device (cpu or gpu)
    :param nn_input_size: input size for the model
    :param window_sizes:
    :param step_sizes:
    :param verbose: 0 if print statements should not be shown, 1 if they should
    :return: array containing the class ids
    """

    # define helper for printing
    printer = Printer(verbose=verbose)

    # cut the input into several, overlapping images
    cuts, start_points = [], []
    for w_s, s_s in zip(window_sizes, step_sizes):
        cts, pts = _cut_windows(image, window_size=w_s, step_size=s_s)
        cuts += cts
        start_points += pts

    printer("Cut image into {} pieces".format(len(cuts)))

    # create array that combines the output predictions
    # we don't know the class count now, so this is just a placeholder
    combined_output = None

    pbar = tqdm(zip(cuts, start_points))

    for cut_image, start_point in pbar:
        output = predict_image(cut_image, model, device, nn_input_size)

        # we didn't know the number of classes before prediction, so create output array now
        if combined_output is None:
            combined_output = np.zeros((image.shape[:2]) + (output.shape[-1],))

        start_x, end_x = start_point[0], start_point[0] + cut_image.shape[1]
        start_y, end_y = start_point[1], start_point[1] + cut_image.shape[0]

        # each of the overlaps creates "votes" for its corresponding pixel which we add up
        combined_output[start_y: end_y, start_x:end_x] += output

    class_id_mask = np.argmax(combined_output, axis=-1)

    return class_id_mask


def predict_image(image, model, device, nn_input_size):
    """

    :param image:
    :param model:
    :param device:
    :param nn_input_size:
    :return:
    """
    # create transformations needed to preprocess image to go into the neural network

    transformations = transforms.Compose([
        Normalize(),
        Resize(nn_input_size),
        ToTensor()
    ])

    sample = {STR.NN_INPUT: image}

    # apply transformations
    sample = transformations(sample)
    nn_input = sample[STR.NN_INPUT]

    nn_input = nn_input.to(device)

    # predict input
    model.eval()
    output = model(nn_input)
    output = output.detach().cpu().numpy()
    output = output[0]  # remove the first dimension which corresponds to the index in the batch
    output = output.transpose(1, 2, 0)

    # scale output up to original size
    factor = image.shape[0] / output.shape[0]
    output = zoom(output, [factor, factor, 1], order=0)

    return output


def _cut_windows(image, window_size, step_size=None):
    """
    Cut an image into several, equally sized windows. step size determines the overlap of the windows.
    :param image: image to be cut
    :param window_size: size of the square windows
    :param step_size: step size between windows, determines overlap. Depending on the image size, window size and
     step size it may not be possible to ensure the given step size since a constant window size is preferred
    :return: list of cut images and list of original, upper left corner points (x, y)
    """
    step_size = int(window_size / 2) if step_size is None else step_size

    h, w = image.shape[:2]
    cuts = []
    start_points = []

    for x in range(0, w - step_size, step_size):
        end_x = np.min([x + window_size, w])
        start_x = end_x - window_size

        # stop if the current rectangle has been done before
        if len(start_points) > 0 and start_x == start_points[-1][0]:
            break

        for y in range(0, h - step_size, step_size):
            end_y = np.min([y + window_size, h])
            start_y = end_y - window_size

            # stop if the current rectangle has been done before
            if len(start_points) > 0 and start_y == start_points[-1][1]:
                break

            cuts.append(image[start_y:end_y, start_x:end_x])
            start_points.append([start_x, start_y])

            # print("x: {}/ y:{} to x: {}/ y:{}".format(start_x, start_y, end_x, end_y))

    return cuts, start_points
