import json
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from coral_reef.constants import paths


def display_class_distribution(stats_file_paths, include_background=True, subtitles=None, title=None,
                               out_file_path=None):
    """
    display a bar plot showing the class distribution
    :param stats_file_paths: list of file paths to the json files containing the class statistics
    :param include_background: True if the background class should be included
    :param subtitles: list of strings - titles for the subplots
    :title: title of the plot
    :param out_file_path: path to the location where the image of the plot should be saved. If None, plot will be
    displayed
    :return: Nothing
    """
    row_counts = len(stats_file_paths)
    plt.figure(figsize=(10, 2.2 * row_counts))
    if title:
        plt.title(title)

    for i, stats_file_path in enumerate(stats_file_paths):
        # read data stats
        with open(stats_file_path, "r") as fp:
            class_stats = json.load(fp)

        classes = list(sorted(class_stats.keys()))
        if not include_background:
            classes.remove("background")

        # get shares for each class
        total = sum([class_stats[c]["share"] for c in classes])
        counts = {c: class_stats[c]["share"] / total * 100 for c in classes}  # make them add up to 100

        ax = plt.subplot(row_counts, 1, i + 1)

        # plot subtitles
        # done like this since it saves space
        if subtitles:
            plt.text(len(classes) / 2, max([counts[c] for c in classes]) * 0.9, subtitles[i], ha="center",
                     size="x-large")
        # plot
        plt.bar(x=np.arange(len(classes)),
                height=[counts[c] for c in classes])

        # only display x ticks for the lowest row
        if i == row_counts - 1:
            plt.xticks(np.arange(len(classes)), classes, rotation=60)
        else:
            plt.xticks([])

        plt.yticks([])
        # ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}%"))

        # add text information
        for x, c in enumerate(classes):
            if counts[c] == max([counts[c] for c in classes]):
                plt.text(x, counts[c] * 0.9, "{:.1f}%".format(counts[c]), ha="center", color="w")
            else:
                if counts[c] > 0.01:
                    plt.text(x, counts[c], "{:.2f}%".format(counts[c]), ha="center")
                else:
                    plt.text(x, counts[c], "{:.3f}%".format(counts[c]), ha="center")

        plt.tight_layout()

    # save or show the plot
    if out_file_path:
        plt.savefig(out_file_path)
    else:
        plt.show()


if __name__ == "__main__":
    stats_file_paths = [os.path.join(paths.DATA_FOLDER_PATH, "class_stats" + d + ".json") for d in
                        ["", "_train", "_valid"]]
    subtitles = ["overall", "training", "validation"]
    out_file_path = os.path.join(paths.PROJECT_FOLDER_PATH, "documentation", "class_distributions.png")
    display_class_distribution(stats_file_paths,
                               include_background=True,
                               title="Class distributions",
                               subtitles=subtitles,
                               out_file_path=out_file_path)
