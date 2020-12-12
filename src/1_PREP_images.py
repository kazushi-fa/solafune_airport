import glob
import numpy as np
from PIL import Image
from natsort import natsorted
import pandas as pd
import configparser

def make_image_prep(files, image_size, dir_out, filename_out):
    images = np.zeros((len(files),image_size,image_size,3))
    i=0
    for file in files:
        img = np.array(Image.open(file).resize((image_size, image_size)))
        images[i,:,:,:] = img
        i=i+1
    np.save(dir_out + filename_out, images.astype('int'))

def make_anno_prep(csv_anno, dir_out, filename_out):
    df = pd.read_csv(csv_anno, header=None)
    anno = df[1].values
    np.save(dir_out + filename_out, anno.astype('int'))

if __name__ == '__main__':
    # SETTING
    ini = configparser.ConfigParser()
    ini.read('./config.ini', 'UTF-8')
    image_size = int(ini['common']['image_size'])
    dir_data = str(ini['Prep']['dir_data'])
    dir_prep = str(ini['Prep']['dir_prep'])
    dir_ori_data = str(ini['common']['dir_ori_data'])
    dir_train_image = dir_data + str(ini['common']['dir_train_image'])
    train_anno = dir_data + str(ini['common']['train_anno'])
    dir_test_image = dir_data + str(ini['common']['dir_test_image'])
    test_anno = dir_data + str(ini['common']['test_anno'])
    dir_eval_image = dir_ori_data + str(ini['common']['dir_eval_image'])

    print("TRAIN")
    files_train_images = natsorted(glob.glob(dir_train_image + "*.jpg"))
    make_image_prep(files_train_images,
                    image_size,
                    dir_prep,
                    "train_images")
    make_anno_prep(train_anno,
                    dir_prep,
                    "train_anno")

    print("TEST")
    files_test_images = natsorted(glob.glob(dir_test_image + "*.jpg"))
    make_image_prep(files_test_images,
                    image_size,
                    dir_prep,
                    "test_images")
    make_anno_prep(test_anno,
                    dir_prep,
                    "test_anno")

    print("EVALUATION")
    files_eval_images = natsorted(glob.glob(dir_eval_image + "*.jpg"))
    make_image_prep(files_eval_images,
                    image_size,
                    dir_prep,
                    "eval_images")

