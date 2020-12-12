import glob
import numpy as np
from PIL import Image
from natsort import natsorted
import pandas as pd
import cv2
import sklearn
from sklearn.cluster import KMeans
import configparser

def make_image_stats_prep(files, dir_out, filename_out):
    images_stats = np.zeros((len(files), 26))
    i=0
    for file in files:
        image_size = 128
        img = np.array(Image.open(file))

        images_stats[i,0] = np.mean(img[:,:,0].reshape(-1))/255
        images_stats[i,1] = np.median(img[:,:,0].reshape(-1))/255
        images_stats[i,2] = np.std(img[:,:,0].reshape(-1))/100
        images_stats[i,3] = np.var(img[:,:,0].reshape(-1))/10000
        images_stats[i,4] = np.max(img[:,:,0].reshape(-1))/255
        
        images_stats[i,5] = np.mean(img[:,:,1].reshape(-1))/255
        images_stats[i,6] = np.median(img[:,:,1].reshape(-1))/255
        images_stats[i,7] = np.std(img[:,:,1].reshape(-1))/100
        images_stats[i,8] = np.var(img[:,:,1].reshape(-1))/10000
        images_stats[i,9] = np.max(img[:,:,1].reshape(-1))/255

        images_stats[i,10] = np.mean(img[:,:,2].reshape(-1))/255
        images_stats[i,12] = np.median(img[:,:,2].reshape(-1))/255
        images_stats[i,13] = np.std(img[:,:,2].reshape(-1))/100
        images_stats[i,14] = np.var(img[:,:,2].reshape(-1))/10000
        images_stats[i,15] = np.max(img[:,:,2].reshape(-1))/255

        img_resized = np.array(Image.open(file).resize((image_size, image_size)))
        image_vector = img_resized.reshape(image_size*image_size, 3)
        cluster = KMeans(n_clusters=3)
        cluster.fit(X=image_vector)

        images_stats[i,16:19] = cluster.cluster_centers_[:,0]/255
        images_stats[i,19:22] = cluster.cluster_centers_[:,1]/255
        images_stats[i,22:25] = cluster.cluster_centers_[:,2]/255

        i=i+1
    np.save(dir_out + filename_out, images_stats)

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
    make_image_stats_prep(files_train_images,
                            dir_prep,
                            "train_images_stats")
    make_anno_prep(train_anno,
                    dir_prep,
                    "train_anno")

    print("TEST")
    files_test_images = natsorted(glob.glob(dir_test_image + "*.jpg"))
    make_image_stats_prep(files_test_images,
                            dir_prep,
                            "test_images_stats")
    make_anno_prep(test_anno,
                            dir_prep,
                            "test_anno")

    print("EVALUATION")
    files_eval_images = natsorted(glob.glob(dir_eval_image + "*.jpg"))
    make_image_stats_prep(files_eval_images,
                            dir_prep,
                            "eval_images_stats")

