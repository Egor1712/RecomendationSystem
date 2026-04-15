from pathlib import Path
import os.path

absolute_path = Path(__file__).resolve().parent

DATASET_DIRECTORY = os.path.join(absolute_path, r'../dataset')
WORKING_DATASET_DIRECTORY = os.path.join(absolute_path, r'../dataset_reduced')
IMAGES_DATASET_DIRECTORY = os.path.join(WORKING_DATASET_DIRECTORY, 'images')

TWO_TOWER_MODEL = 'two_tower'
LIGTFM = 'lightfm'
FOLDS = 'FOLDS'