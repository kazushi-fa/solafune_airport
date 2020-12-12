import glob
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from natsort import natsorted
import pandas as pd
import random
import configparser

def make_image_prep_aug(files, dir_anno, out_path, g_num, image_size, dir_out, filename_out):
    i=0
    df = pd.read_csv(dir_anno, header=None)
    file_name=[]
    anno = df[1].values
    anno_out = np.zeros((anno.shape[0]*g_num,1))
    for file in files:
        print(file)
        img = Image.open(file).resize((image_size, image_size))
        img.save(out_path + 'image_enhance'+'_'+str(i)+"_0"+'.jpg', quality=95)

        for j in range(g_num):
            anno_out[(i*g_num+j)] = anno[i]
            if j == 0:
                im_enhance = img
            elif j % 4 == 0:
                im_enhance = random_image_creation(img,0)
            elif j % 4 == 1:
                im_enhance = random_image_creation(img,1)
            elif j % 4 == 2:
                im_enhance = random_image_creation(img,2)
            else:
                im_enhance = random_image_creation(img,3)
            im_enhance.save(out_path + 'image_enhance'+'_'+str(i)+"_" + str(j) + '.jpg', quality=95)
            file_name.append('image_enhance'+'_'+str(i)+"_" + str(j) + '.jpg')
        i=i+1
    anno_out_stack = np.stack([np.array(file_name), anno_out.reshape(-1)], 1)
    df_anno_out = pd.DataFrame(anno_out_stack)
    df_anno_out.to_csv(dir_out + filename_out + ".csv", index=False, header=False, encoding="utf-8")
    #np.savetxt(dir_out + filename_out + ".csv", anno_out.astype('int'))

def random_image_creation(img,flag):
        random_1 = random.uniform(0.5, 1.5)
        enhancer_1 = ImageEnhance.Color(img) 
        im_enhance_1 = enhancer_1.enhance(random_1)

        random_2 = random.uniform(0.5, 1.5)
        enhancer_2 = ImageEnhance.Contrast(im_enhance_1)
        im_enhance_2 = enhancer_2.enhance(random_2)

        random_3 = random.uniform(0.5, 1.5)
        enhancer_3 = ImageEnhance.Brightness(im_enhance_2)
        im_enhance_3 = enhancer_3.enhance(random_3)

        random_4 = random.uniform(0.5, 1.5)
        enhancer_4 = ImageEnhance.Sharpness(im_enhance_3)
        im_enhance_4 = enhancer_4.enhance(random_4)

        if flag == 0:
            return im_enhance_4.rotate(90)
        elif flag == 1:
            return im_enhance_4.rotate(180)
        elif flag == 2:
            out = im_enhance_4
            return out
        else:
            out = im_enhance_4.rotate(270)
            return out

def make_image_prep(files, image_size, dir_out, filename_out):
    images = np.zeros((len(files),image_size,image_size,3))
    i=0
    for file in files:
        img = np.array(Image.open(file).resize((image_size, image_size)))
        images[i,:,:,:] = img
        i=i+1
    np.save(dir_out + filename_out, images.astype('int'))

def make_anno_prep_aug(csv_anno, dir_out, filename_out):
    df = pd.read_csv(csv_anno, header=None)
    anno = df[1].values
    anno_out = np.zeros((anno.shape[0]*4,1))
    for i in range(anno.shape[0]):
        anno_out[(i*4+0),:] = anno[i]
        anno_out[(i*4+1),:] = anno[i]
        anno_out[(i*4+2),:] = anno[i]
        anno_out[(i*4+3),:] = anno[i]
    np.save(dir_out + filename_out, anno_out.astype('int'))

def make_anno_prep(csv_anno, dir_out, filename_out):
    df = pd.read_csv(csv_anno, header=None)
    anno = df[1].values
    np.save(dir_out + filename_out, anno.astype('int'))

if __name__ == '__main__':
    # SETTING
    ini = configparser.ConfigParser()
    ini.read('./config.ini', 'UTF-8')
    dir_data = str(ini['DataAug']['dir_data'])
    dir_prep = str(ini['DataAug']['dir_prep'])
    out_path = str(ini['DataAug']['out_path'])
    image_size = int(ini['common']['image_size'])
    dir_train_image = dir_data + str(ini['common']['dir_train_image'])
    dir_test_image = dir_data + str(ini['common']['dir_test_image'])
    dir_eval_image = dir_data + str(ini['common']['dir_eval_image'])
    out_path_train =  out_path + str(ini['common']['dir_train_image'])
    out_path_test =  out_path + str(ini['common']['dir_test_image'])
    train_anno = dir_data + str(ini['common']['train_anno'])
    test_anno = dir_data + str(ini['common']['test_anno'])
    aug_train = int(ini['DataAug']['aug_train'])
    aug_test = int(ini['DataAug']['aug_test'])

    print("TRAIN")
    files_train_images = natsorted(glob.glob(dir_train_image + "*.jpg"))
    make_image_prep_aug(files_train_images,
                        train_anno,
                        out_path_train,
                        aug_train,
                        image_size,
                        out_path,
                        "traindataset_anotated")

    print("TEST")
    files_test_images = natsorted(glob.glob(dir_test_image + "*.jpg"))
    make_image_prep_aug(files_test_images,
                        test_anno,
                        out_path_test,
                        aug_test,
                        image_size,
                        out_path,
                        "testdataset_anotated")

