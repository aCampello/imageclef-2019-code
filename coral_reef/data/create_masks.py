import os
import cv2
import json
import numpy as np
from pprint import pprint

from coral_reef.constants import paths


def parse_csv_data_file(csv_file_path):
    """
    Read data file and return a dict containing the data
    :param csv_file_path: file containing the data
    :return: dict containing the annotation data
    """
    data = {}

    with open(csv_file_path, "r") as file:
        for line in file:
            if not line.strip():
                break

            values = line.split(" ")
            image_name = values[0] + ".JPG"

            coordinates = values[4:]
            if not len(coordinates) % 2 == 0:
                Warning("Coordinates have an uneven count for {}. skipping".format(image_name))
                continue

            annotation = {
                "class": values[2],
                "coordinates": [[int(coordinates[i]), int(coordinates[i + 1])] for i in range(0, len(coordinates), 2)]
            }

            data.setdefault(image_name, []).append(annotation)

    return data


def create_annotation_masks():
    """
    Create mask images acting as annotations from the annotations data file. Mask files will be stored in the
    path.MASK_FOLDER_PATH folder
    :return: Nothing
    """

    # parse the annotations file to get the data
    csv_file_path = os.path.join(paths.DATA_FOLDER_PATH, "annotations.csv")
    data = parse_csv_data_file(csv_file_path)

    # create a list containing all classes
    classes = list(sorted(set([annotation["class"] for img_name in data.keys() for annotation in data[img_name]])))

    # create a colour for each class, 0 is background
    colours = np.linspace(0, 255, len(classes) + 1).astype(int).tolist()

    class_mapping = {c: i + 1 for i, c in enumerate(classes)}

    colour_mapping = {"background": 0}
    colour_mapping.update({c: colours[class_mapping[c]] for c in classes})

    for i, image_name in enumerate(data.keys()):
        print("processing image {} of {}".format(i + 1, len(data.keys())))

        # create mask based on the size of the corresponding image
        image_path = os.path.join(paths.IMAGE_FOLDER_PATH, image_name)
        image_height, image_width = cv2.imread(image_path).shape[:2]
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # go through each annotation entry and fill the corresponding polygon. Color corresponds to class
        for annotation in data[image_name]:
            colour = colour_mapping[annotation["class"]]

            points = annotation["coordinates"]
            cv2.fillPoly(mask, [np.array(points)], color=colour)

        # save the mask
        name, _ = os.path.splitext(image_name)
        out_name = name + "_mask.png"
        cv2.imwrite(os.path.join(paths.MASK_FOLDER_PATH, out_name), mask)

    # write color mapping to file
    with open(os.path.join(paths.DATA_FOLDER_PATH, "colour_mapping.json"), "w") as fp:
        json.dump(colour_mapping, fp, indent=4)


def correct_masks():
    """
    There are some masks that need to be rotated by 180 degree to fit the data. apparently there has been a problem
    during annotation or so. This method is only useful for the current version (pre 1.4) of the data. The data on the
    server should soon be fixed.
    :return:
    """
    mask_files = ["2018_0729_112409_029_mask.png",
                  "2018_0729_112442_045_mask.png",
                  "2018_0729_112537_066_mask.png",
                  "2018_0729_112536_063_mask.png",
                  "2018_0729_112541_053_mask.png",
                  "2018_0729_112455_038_mask.png",
                  "2018_0729_112449_036_mask.png"]

    for mask_file in mask_files:
        path = os.path.join(paths.MASK_FOLDER_PATH, mask_file)
        mask = cv2.imread(path)
        mask = mask[::-1, ::-1, 0]
        cv2.imwrite(path, mask)

if __name__ == "__main__":
    create_annotation_masks()

# def create_data_files():
#
#     data = []
#     for image_file in image_files:
#         _, image_name = os.path.split(image_file)
#         image_base_name, _ = os.path.splitext(image_name)
#         mask_name = image_base_name + "_mask.png"
#         data.append({
#             "image_name": "images/" + image_name,
#             "mask_name": "masks/" + mask_name
#         })
#
