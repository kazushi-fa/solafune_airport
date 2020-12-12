from keras.models import Sequential
from keras.optimizers import SGD,adam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
import glob
import pathlib
from keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import regularizers
import tensorflow as tf
import configparser

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    ratio = 0.5
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu',
                kernel_initializer='he_normal',
                bias_initializer='zeros'))
    model.add(Dropout(ratio))
    model.add(Dense(4096, activation='relu',
                kernel_initializer='he_normal',
                bias_initializer='zeros'))
    model.add(Dropout(ratio))
    model.add(Dense(1000, activation='relu',
                kernel_initializer='he_normal',
                bias_initializer='zeros'))
    model.add(Dropout(ratio))
    model.add(Dense(1, activation='sigmoid',
                kernel_initializer='he_normal',
                bias_initializer='zeros'))

    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss=custom_loss)
    model.compile(optimizer='adam', loss=custom_loss)

    return model

def custom_loss(y_true, y_pred):
    normalize_num = 80000000
    y_true = y_true * normalize_num
    y_pred = y_pred * normalize_num
    out = tf.square(tf.log(y_true + 1) - tf.log(y_pred + 1))
    return out

def plot_history_loss(history,axL):
    axL.plot(history['loss'],label="loss for training")
    axL.plot(history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

def calc_RMSLE(Y_train, Y_pred):
    RMSLE = np.square(np.log(Y_train + 1) - np.log(Y_pred + 1))
    return RMSLE

def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data)
        while True:
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X = shuffled_data[start_index: end_index]
                y = shuffled_labels[start_index: end_index]
                yield X, y

    return num_batches_per_epoch, data_generator()

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

if __name__ == '__main__':
    channel = 3
    num_classes = 1

    # SETTING
    ini = configparser.ConfigParser()
    ini.read('./config.ini', 'UTF-8')
    image_size = int(ini['common']['image_size'])
    img_rows, img_cols = image_size, image_size
    batch_size = int(ini['Train']['batch_size'])
    nb_epoch = int(ini['Train']['nb_epoch'])
    normalize_num = int(ini['Train']['normalize_num'])
    dir_prep = str(ini['Train']['dir_prep'])
    dir_result = str(ini['Train']['dir_result_VGG-like'])
    dir_data = str(ini['Train']['dir_data'])
    dir_tflog = str(ini['Train']['dir_tflog'])
    dir_eval_image = str(ini['common']['dir_ori_data']) + str(ini['common']['dir_eval_image'])

    # データのロード
    X_train_temp = np.load(dir_prep + 'train_images.npy', allow_pickle=True)/255
    Y_train_temp = np.load(dir_prep + 'train_anno.npy', allow_pickle=True)/normalize_num
    X_valid_temp = np.load(dir_prep + 'test_images.npy', allow_pickle=True)/255
    Y_valid_temp = np.load(dir_prep + 'test_anno.npy', allow_pickle=True)/normalize_num

    # データのシャッフル
    all_data = np.concatenate([X_train_temp, X_valid_temp], axis=0)
    all_label = np.concatenate([Y_train_temp,Y_valid_temp], axis=0)
    num_train = X_train_temp.shape[0]
    num_valid  = X_valid_temp.shape[0]
    num_all = num_train + num_valid
    print(num_train, num_valid, num_all, all_data.shape, all_label.shape)
    print(Y_train_temp.shape, Y_valid_temp.shape)
    del X_train_temp,Y_train_temp,X_valid_temp,Y_valid_temp

    id_all   = np.random.choice(num_all, num_all, replace=False)
    id_train  = id_all[:num_train]
    id_valid = id_all[num_train:]

    X_train = all_data[id_train]
    Y_train = all_label[id_train]
    X_valid = all_data[id_valid]
    Y_valid = all_label[id_valid]

    X_eval = np.load(dir_prep + 'eval_images.npy', allow_pickle=True)/255

    print("!!!!",X_train.shape,Y_train.shape,X_valid.shape,Y_valid.shape,X_eval.shape)
    print("!!!!",all_data[id_train].shape)

    # モデルのロード
    model = vgg16_model(img_rows, img_cols, channel, num_classes)

    # モデルの学習
    es_cb = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')
    cp = ModelCheckpoint(dir_result + "best.hdf5", monitor="val_loss", verbose=1,
                     save_best_only=True, save_weights_only=True)
    tb_cb = TensorBoard(log_dir=dir_tflog, histogram_freq=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              callbacks=[cp, es_cb, reduce_lr, tb_cb]
              )

    train_steps, train_batches = batch_iter(X_train, Y_train, batch_size)
    valid_steps, valid_batches = batch_iter(X_valid, Y_valid, batch_size)
    model.fit_generator(train_batches, train_steps,
                    epochs=nb_epoch, 
                    validation_data=valid_batches,
                    validation_steps=valid_steps,
                    callbacks=[cp, es_cb, reduce_lr, tb_cb]
                    )

    model.save_weights(dir_result + 'param.hdf5')
    with open(dir_result + 'history.json', 'w') as f:
        json.dump(history.history, f, cls = MyEncoder)

    #ログの書き出し
    f = open(dir_result + 'history.json', 'r')
    history = json.load(f)
    f.close()
    fig, (axL) = plt.subplots(ncols=1, figsize=(10,4))
    plot_history_loss(history, axL)
    fig.savefig(dir_result + 'loss.png')
    plt.close()

    # 学習結果のロード
    model.load_weights(dir_result + "best.hdf5")

    # trainデータの出力
    Y_train = Y_train * normalize_num
    train_pred = model.predict(X_train, batch_size=batch_size, verbose=1).reshape(-1) * normalize_num
    RMSLE_train_cal = calc_RMSLE(Y_train, train_pred)
    train = np.stack([Y_train, train_pred, RMSLE_train_cal])
    df_train = pd.DataFrame(train.T, columns=['TRUE', 'MODEL', 'RMSLE_cal'])
    df_train.to_csv(dir_result + 'train.csv')

    # valデータの出力
    Y_valid = Y_valid * normalize_num
    valids_pred = model.predict(X_valid, batch_size=batch_size, verbose=1).reshape(-1) * normalize_num
    RMSLE_cal = calc_RMSLE(Y_valid, valids_pred)
    valids = np.stack([Y_valid, valids_pred, RMSLE_cal])
    df_valids = pd.DataFrame(valids.T, columns=['TRUE', 'MODEL', 'RMSLE_cal'])
    df_valids.to_csv(dir_result + 'valids.csv')

    RMSLE = np.sum(df_valids['RMSLE_cal'].values)/len(df_valids)
    np.savetxt(dir_result + 'RMSLE.txt', RMSLE.reshape(-1))
    print("Val RMSLE : ", RMSLE)

    # evalデータの出力
    files_eval_images = natsorted(glob.glob(dir_eval_image + "*.jpg"))
    file_name=[]
    i=0
    for file in files_eval_images:
        file_name.append(file.replace(dir_eval_image, ""))
        i=i+1
    predictions = model.predict(X_eval, batch_size=batch_size, verbose=1).reshape(-1) * normalize_num
    predictions = (predictions).astype(np.int32)
    predictions_arr = np.stack([np.array(file_name), predictions], 1)
    df_predictions = pd.DataFrame(predictions_arr)
    df_predictions.to_csv(dir_result + 'predictions.csv', index=False, header=False, encoding="utf-8")
    