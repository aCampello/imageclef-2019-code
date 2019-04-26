import glob
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from coral_reef.constants import mapping, paths
from coral_reef.ml.utils import class_id_mask_to_colour_mask
from coral_reef.visualisation import visualisation


def _get_contour(mask):
    _, contours, _ = cv2.findContours(mask, 1, 2)
    cnt = contours[0]
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    return approx


def show_polygon(image_shape, points):
    image = np.zeros(image_shape).astype(np.uint8)

    image = cv2.fillPoly(image, [points], True, 255)
    plt.imshow(image)
    plt.show()


def create_submission_file(prediction_file_names, output_file_path):
    """
    Creates a submission file in accordance with the rules of the competition based on the given files
    :param prediction_file_names: paths to numpy files or arrays with size (h, w, class_count)
    :param output_file_path: path to the  file where the results will be stored
    :return: Nothing
    """

    colour_mapping = mapping.get_colour_mapping()
    class_names = sorted(colour_mapping.keys())

    out_lines = []

    for i, file in enumerate(prediction_file_names):
        print("file {} of {}\n".format(i + 1, len(prediction_file_names)))

        prediction = np.load(file)

        # start output row
        line = "{};".format(os.path.split(file)[1][:-4])

        class_id_mask = np.argmax(prediction, axis=-1)

        class_ids = np.unique(class_id_mask).tolist()

        # remove background class
        if 0 in class_ids:
            class_ids.remove(0)

        pbar = tqdm(class_ids, ncols=50)
        for class_id in pbar:
            mask = ((class_id_mask == class_id) * 255).astype(np.uint8)

            # apply opening to mask to remove small coral-areas
            # TODO: the size needs to be set to something sensible!
            kernel = np.ones((21, 21), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # find connected components
            ret, labels = cv2.connectedComponents(mask)
            for label in range(1, ret):  # label 0 is for background of the connected components
                cc_mask = ((labels == label) * 1)
                polygon = _get_contour((cc_mask * 255).astype(np.uint8))
                confidence = (prediction[:, :, class_id] * cc_mask).sum() / cc_mask.sum()
                # print("confidence: {:.2f}".format(confidence))

                line += "{} {:.2f}:".format(class_names[class_id], confidence)
                line += _create_polygon_string(polygon)

        out_lines.append(line)

    with open(output_file_path, "w") as fp:
        fp.writelines(out_lines)


def _create_polygon_string(polygon):
    out_string = ""
    polygon = polygon.squeeze()
    for pt in polygon:
        out_string += "{}+{}+".format(int(pt[0]), int(pt[1]))
    out_string = out_string[:-1]  # remove last '+'
    return out_string


if __name__ == '__main__':
    predictions_folder_path = os.path.join(paths.DATA_FOLDER_PATH, "predictions")
    files = glob.glob(os.path.join(predictions_folder_path, "*.npy"))

    out_file_path = "/home/aljo/filament/coral_reef/data/out/predictions.txt"

    create_submission_file(files, out_file_path)
