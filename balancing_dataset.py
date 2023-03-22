from const import Config
import os
import pandas as pd
from pathlib import Path


def move_spoof_to_val(img_dir_train: Path, img_dir_val: Path, csv_train_path: Path, csv_val_path: Path, verbose=False):
    """
    Move images with label=1 to Val folder in order
    to have equal representation of each class in Train folder
    """

    counter = 0
    image_to_label = {}
    csv_train = pd.read_csv(csv_train_path)

    # remove path part to jpg from csv column ['image']
    filename_train = os.path.basename(csv_train_path).split('.')[0]
    csv_train['image'] = csv_train['image'].str.replace(f'celeba/{filename_train}/', '')

    # create dict Image: Label
    for _, row in csv_train.iterrows():
        image_name = row['image']
        label = row['spoof_label']
        image_to_label[image_name] = label
    # move files from train to val directory
    for file in os.listdir(img_dir_train):
        if image_to_label[file] == 1 and counter < 860:
            os.remove(Path(img_dir_train, f'{file}'))
            counter += 1

    if verbose:
        print('Train:', len(os.listdir(img_dir_train)), 'Val:', len(os.listdir(img_dir_val)))


if __name__ == '__main__':

    move_spoof_to_val(Config.IMG_DIR_TRAIN, Config.IMG_DIR_VAL, Config.CSV_TRAIN, Config.CSV_VAL, verbose=True)

