import json
import os

import numpy as np

import matplotlib.pyplot as plt

from coral_reef.constants import paths


def display_class_distribution(include_background=True):
    """
    display a bar plot showing the class distribution
    :param include_background: True if the background class should be included
    :return: Nothing
    """

    # read data stats
    stats_file_path = os.path.join(paths.DATA_FOLDER_PATH, "class_stats.json")
    with open(stats_file_path, "r") as fp:
        class_stats = json.load(fp)

    classes = list(sorted(class_stats.keys()))
    if not include_background:
        classes.remove("background")

    # get shares for each class
    total = sum([class_stats[c]["share"] for c in classes])
    counts = {c: class_stats[c]["share"] / total for c in classes}  # make them add up to 1

    # plot
    plt.bar(x=np.arange(len(classes)),
            height=[counts[c] for c in classes])
    plt.xticks(np.arange(len(classes)), classes, rotation=45)

    # add text information
    for x, c in enumerate(classes):
        plt.text(x, counts[c], "{:.1f}%".format(counts[c] * 100), ha="center")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    display_class_distribution(include_background=False)
