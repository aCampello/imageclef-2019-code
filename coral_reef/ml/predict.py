import json
import os
import sys

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import zoom
import cv2

from coral_reef.ml.data_set import Normalize, Resize, ToTensor
from coral_reef.constants import paths
from coral_reef.constants import strings as STR

from coral_reef.ml.utils import load_state_dict, cut_windows
from coral_reef.utils.print_utils import Printer

sys.path.extend([paths.DEEPLAB_FOLDER_PATH, os.path.join(paths.DEEPLAB_FOLDER_PATH, "utils")])
from modeling.deeplab import DeepLab


def _predict_by_cutting(image, model, device, nn_input_size, window_sizes, step_sizes, verbose=0):
    """
    Predicts the given image with the given model. Image is cut into several overlapping pieces which will be predicted
    individually. Will be recombined after prediction.
    :param image: image as numpy array as uint8
    :param model: model used for prediction
    :param device: PyTorch device (cpu or gpu)
    :param nn_input_size: input size for the model
    :param window_sizes: list of sizes that determine how the image will be cut (for sliding window). Image will be cut
    into squares
    :param step_sizes: list of step sizes for sliding window
    :param verbose: 0 if print statements should not be shown, 1 if they should
    :return: array containing the class ids
    """

    # define helper for printing
    printer = Printer(verbose=verbose)

    # cut the input into several, overlapping images
    cuts, start_points = [], []
    for w_s, s_s in zip(window_sizes, step_sizes):
        cts, pts = cut_windows(image, window_size=w_s, step_size=s_s)
        cuts += cts
        start_points += pts

    printer("Cut image into {} pieces".format(len(cuts)))

    # create array that combines the output predictions
    # we don't know the class count now, so this is just a placeholder
    combined_output = None
    counts = None  # this will be used to convert the combined output later into scores/confidences again

    for cut_image, start_point in zip(cuts, start_points):

        output = _predict_image(cut_image, model, device, nn_input_size)

        # we didn't know the number of classes before prediction, so create output array now
        if combined_output is None:
            combined_output = np.zeros((image.shape[:2]) + (output.shape[-1],))
            counts = np.zeros((image.shape[:2]) + (output.shape[-1],))

        start_x, end_x = start_point[0], start_point[0] + cut_image.shape[1]
        start_y, end_y = start_point[1], start_point[1] + cut_image.shape[0]

        # each of the overlaps creates "votes" for its corresponding pixel which we add up
        combined_output[start_y: end_y, start_x:end_x] += output

        # store information how often each pixel got votes so that we can calculate the average later
        counts[start_y: end_y, start_x:end_x] += 1

    # turn it back into confidences
    combined_output /= counts

    return combined_output


def _predict_image(image, model, device, nn_input_size):
    """

    :param image: image to predict
    :param model: model for prediction
    :param device: PyTorch device (cpu or gpu)
    :param nn_input_size: shape that the image will be scaled to
    :return: nn output
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


def predict(image_file_paths, model, nn_input_size, res_fcn=None, window_sizes=None, step_sizes=None, device=None):
    """
    Segment images
    :param image_file_paths: paths to images that will be predicted
    :param model: model used for prediction
    :param nn_input_size: size that the images will be scaled to before feeding them into the nn
    :param res_fcn: custom method that will be called with the result prediction mask and the image index. If None, data
    will be returned in a list
    :param window_sizes: list of sizes that determine how the image will be cut (for sliding window). Image will be cut
    into squares
    :param step_sizes: list of step sizes for sliding window
    :param device: PyTorch device (cpu or gpu)
    :return: list of prediction masks if res_fcn is None, else Nothing
    """

    device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prediction_results = []

    pbar = tqdm(image_file_paths, ncols=50)

    for i, image_file_path in enumerate(pbar):
        # print("predicting image {} of {}".format(i + 1, len(image_file_paths)))

        image = cv2.imread(image_file_path)[:, :, ::-1]

        window_sizes = window_sizes if window_sizes is not None else [500, 1000, 1500]
        step_sizes = step_sizes if step_sizes is not None else [350, 750, 1000]

        prediction = _predict_by_cutting(image=image,
                                         model=model,
                                         device=device,
                                         nn_input_size=nn_input_size,
                                         window_sizes=window_sizes,
                                         step_sizes=step_sizes,
                                         verbose=0)

        # deal with the results. Either apply custom function or append to list
        if res_fcn is not None:
            res_fcn(prediction, i)
        else:
            prediction_results.append(prediction)

    if res_fcn is None:
        return prediction_results
