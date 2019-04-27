import os
import json
from random import shuffle

from coral_reef.ml import train
from coral_reef.constants import strings as STR
from coral_reef.constants import paths

nn_input_size = 512
output_stride = 16
backbone = "drn"
if backbone == "drn":
    output_stride = 8  # will be set anyway within the deeplab model
multi_gpu = True


def _get_instructions_train_normal():
    images_per_batch = int(2 * 1)
    crops_per_image = int(16 * 2)

    instructions = {
        STR.MODEL_NAME: "coral_aws_V1",

        STR.IMAGES_PER_BATCH: images_per_batch,

        STR.CROPS_PER_IMAGE: crops_per_image,

        STR.BATCH_SIZE: int(images_per_batch * crops_per_image),

        STR.EPOCHS: 50,

        STR.NN_INPUT_SIZE: nn_input_size,

        STR.LEARNING_RATE: 1e-3,

        STR.MULTI_GPU: multi_gpu,

        STR.BACKBONE: backbone,

        STR.DEEPLAB_OUTPUT_STRIDE: output_stride,

        # STR.STATE_DICT_FILE_PATH: os.path.join(paths.MODELS_FOLDER_PATH, "coral_V3", "model_best.pth"),

        STR.CLASS_STATS_FILE_PATH: os.path.join(paths.DATA_FOLDER_PATH, "class_stats_train.json"),

        STR.CROP_SIZE_MIN: max([400, nn_input_size]),

        STR.CROP_SIZE_MAX: 1500,

        STR.USE_LR_SCHEDULER: True,

        STR.LOSS_WEIGHT_MODIFIER: 1.025

    }

    return instructions


def _get_instructions_train_hard():
    instructions = {
        STR.MODEL_NAME: "coral_V4_hard",

        STR.BATCH_SIZE: 128,

        STR.EPOCHS: 15,

        STR.NN_INPUT_SIZE: nn_input_size,

        STR.LEARNING_RATE: 9 * 1e-4,

        STR.MULTI_GPU: multi_gpu,

        STR.BACKBONE: backbone,

        STR.DEEPLAB_OUTPUT_STRIDE: output_stride,

        STR.STATE_DICT_FILE_PATH: os.path.join(paths.MODELS_FOLDER_PATH, "coral_aws_V1", "model_best.pth"),

        STR.CLASS_STATS_FILE_PATH: os.path.join(paths.DATA_FOLDER_PATH, "class_stats_train_hard.json"),

        STR.USE_LR_SCHEDULER: True,

        STR.LOSS_WEIGHT_MODIFIER: 1.025

    }

    return instructions


def execute_training_normal():
    # print("setting visible device for cuda")
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"

    instructions = _get_instructions_train_normal()

    data_folder_path = paths.DATA_FOLDER_PATH
    image_base_dir = data_folder_path

    with open(os.path.join(data_folder_path, "data_train.json"), "r") as fp:
        data_train = json.load(fp)

    with open(os.path.join(data_folder_path, "data_valid.json"), "r") as fp:
        data_valid = json.load(fp)

    shuffle(data_train)
    shuffle(data_valid)

    train.train(data_train=data_train,
                data_valid=data_valid,
                image_base_dir=image_base_dir,
                instructions=instructions)


def execute_training_hard():
    # print("setting visible device for cuda")
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"

    instructions = _get_instructions_train_hard()

    data_folder_path = paths.DATA_FOLDER_PATH
    image_base_dir = data_folder_path

    with open(os.path.join(data_folder_path, "data_train_hard.json"), "r") as fp:
        data_train = json.load(fp)

    with open(os.path.join(data_folder_path, "data_valid.json"), "r") as fp:
        data_valid = json.load(fp)

    shuffle(data_train)
    shuffle(data_valid)

    train.train(data_train=data_train,
                data_valid=data_valid,
                image_base_dir=image_base_dir,
                instructions=instructions)


if __name__ == "__main__":
    execute_training_normal()
