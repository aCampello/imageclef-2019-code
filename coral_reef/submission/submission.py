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

    biggest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            biggest_contour = contour

    cnt = biggest_contour
    epsilon = 0.001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    return approx


def show_polygon(image_shape, points):
    image = np.zeros(image_shape).astype(np.uint8)

    image = cv2.fillPoly(image, [points], color=255)
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

            kernel = np.ones((21, 21), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            line += "{} ".format(class_names[class_id])
            # find connected components
            ret, labels = cv2.connectedComponents(mask)
            for j, label in enumerate(range(1, ret)):  # label 0 is for background of the connected components
                cc_mask = ((labels == label) * 255).astype(np.uint8)

                non_nulls = np.where(cc_mask)

                cv2.floodFill(cc_mask, None, (non_nulls[1][0], non_nulls[0][0]), 255)

                polygon = _get_contour(cc_mask)
                cc_mask = cc_mask / 255
                confidence = (prediction[:, :, class_id] * cc_mask).sum() / cc_mask.sum()
                # print("confidence: {:.2f}".format(confidence))
                if j > 0:
                    line += ","
                line += "{:.2f}:".format(confidence)
                line += _create_polygon_string(polygon)
            line += ";"
        line = line[:-1]  # remove last semicolon
        out_lines.append(line)

    with open(output_file_path, "w") as fp:
        fp.writelines(out_lines)


def parse_submission_line(line):
    data = []

    splits = line.split(";")
    file_id = splits[0]
    for i in range(1, len(splits)):
        class_name, rest = splits[i].split(" ")
        components = rest.split(",")
        for component in components:
            confidence, polygon_string = component.split(":")

            poly_split = polygon_string.split("+")
            polygon = [[int(poly_split[i]), int(poly_split[i + 1])] for i in range(0, len(poly_split), 2)]

            data.append({
                "class_name": class_name,
                "confidence": float(confidence),
                "polygon": polygon
            })

    return file_id, data


def _create_polygon_string(polygon):
    out_string = ""
    polygon = polygon.squeeze()
    for pt in polygon:
        out_string += "{}+{}+".format(int(pt[0]), int(pt[1]))
    out_string = out_string[:-1]  # remove last '+'
    return out_string


def __create_submission():
    predictions_folder_path = os.path.join(paths.DATA_FOLDER_PATH, "predictions")
    files = glob.glob(os.path.join(predictions_folder_path, "*.npy"))

    out_file_path = "/home/aljo/filament/coral_reef/data/out/predictions.txt"

    create_submission_file(files, out_file_path)


def __verify_submission():
    file_path = "/home/aljo/filament/coral_reef/data/out/predictions.txt"
    colour_mapping = mapping.get_colour_mapping()

    with open(file_path, "r") as fp:
        for line in fp:

            # parse file
            file_id, data = parse_submission_line(line)
            confidences = [d["confidence"] for d in data]
            indices = np.argsort(confidences)

            # draw data on image
            colour_mask_1 = np.zeros((3024, 4032)).astype(np.uint8)
            for index in indices:
                polygon = np.array(data[index]["polygon"])
                polygon = polygon.reshape((-1, 1, 2))

                colour_mask_1 = cv2.fillPoly(colour_mask_1, [polygon], color=colour_mapping[data[index]["class_name"]])

            # load original prediciton
            prediction_file = os.path.join(paths.DATA_FOLDER_PATH, "predictions", file_id + ".npy")
            prediction = np.load(prediction_file)
            class_id_mask = np.argmax(prediction, axis=-1)
            colour_mask_2 = class_id_mask_to_colour_mask(class_id_mask)

            plt.subplot(121)
            plt.imshow(colour_mask_1)
            plt.subplot(122)
            plt.imshow(colour_mask_2)
            plt.show()


if __name__ == '__main__':
    __create_submission()
    __verify_submission()
