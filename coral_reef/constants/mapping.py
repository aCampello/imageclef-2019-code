import json
import os

from coral_reef.constants.paths import DATA_FOLDER_PATH


def get_colour_mapping():
    file_path = os.path.join(DATA_FOLDER_PATH, "colour_mapping.json")
    with open(file_path, "r") as fp:
        colour_mapping = json.load(fp)
    return colour_mapping
