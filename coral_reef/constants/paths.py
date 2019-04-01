import os
import pathlib



p = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))

PROJECT_FOLDER_PATH = str(pathlib.Path(*p.parts[:-2]))


DATA_FOLDER_PATH = os.path.abspath(os.path.join(PROJECT_FOLDER_PATH, "data"))
IMAGE_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "images")
MASK_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "masks")
