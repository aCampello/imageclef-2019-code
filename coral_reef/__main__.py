import json
import os

from coral_reef.constants import paths
from coral_reef.ml.train import train
import coral_reef.constants.strings as STR

if __name__ == "__main__":
    # train(data_train, data_valid, image_base_dir, instructions)

    # image_base_dir = A directory with all CLEF images (for training and validating)
    # data_train
    # data_valid
    # instructions: Dict - has to have the following keys:

    instruction_keys = [
        STR.EPOCHS,
        STR.MODEL_NAME,
        STR.NN_INPUT_SIZE,  # TBC
        (STR.STATE_DICT_FILE_PATH, None),  # default = none
        STR.CROPS_PER_IMAGE,
        STR.IMAGES_PER_BATCH,
        STR.BATCH_SIZE,
        (STR.BACKBONE, "resnet"),  # default = resnet
        (STR.DEEPLAB_OUTPUT_STRIDE, 16),  # default = 16
        (STR.LEARNING_RATE, 1e-5),  # default = 1e-5
        (STR.MULTI_GPU, False),  # default = False
        STR.CLASS_STATS_FILE_PATH,
        (STR.USE_LR_SCHEDULER, True)  # default = True
    ]

    # Instructions need the following parameters
    for key in instruction_keys:
        if type(key) == tuple:
            print(f"{key[0]} (default = {key[1]})")
        else:
            print(key)

    instructions = {}  # To be replaced with real instruction file
    image_base_dir = "data"

    with open(os.path.join(paths.DATA_FOLDER_PATH, "data_train.json"), "r") as fp:
        data_train = json.load(fp)

    with open(os.path.join(paths.DATA_FOLDER_PATH, "data_valid.json"), "r") as fp:
        data_valid = json.load(fp)

    train(data_train=data_train,
          data_valid=data_valid,
          image_base_dir=image_base_dir,
          instructions=instructions)
