import os
import six
import time

import numpy as np
import scipy.io as sio

import theano
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD


def load_data():
    MAT_FILE = 'data_new.mat'

    data = sio.loadmat(MAT_FILE)

    train_x = data['train_x'].reshape(-1, 1, 64, 64)
    train_y = data['train_y'].reshape(-1)

    indices = np.arange(len(train_x))
    np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]

    valid_len = len(train_x) / 10

    train_x = train_x / np.float32(256.0)

    train_x, val_x = train_x[:-valid_len], train_x[-valid_len:]
    train_y, val_y = train_y[:-valid_len], train_y[-valid_len:]

    return (train_x, train_y, val_x, val_y)


def build_mlp():
    model = Sequential()

    model.add(Flatten(input_shape=(1, 64, 64)))

    model.add(Dense(256, init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model


def build_cnn():
    model = Sequential()

    # 64 * 64 * 1
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform', input_shape=(1, 64, 64)))
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 32 * 32 * 16

    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 16 * 16 * 32

    model.add(Convolution2D(128, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 8 * 8 * 64

    model.add(Convolution2D(256,  3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 4 * 4 * 128

    model.add(Convolution2D(512, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(Convolution2D(512, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2 * 2 * 256

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(2048, init='he_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, init='he_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def build_cnn_bn():
    model = Sequential()

    # 64 * 64 * 1
    model.add(Convolution2D(8, 5, 5, border_mode='same',
                            init='he_uniform', subsample=(2, 2),
                            input_shape=(1, 64, 64)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # 32 * 32 * 8

    model.add(Convolution2D(8, 3, 3, border_mode='same',
                            init='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 3, 3, border_mode='same',
                            init='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 16 * 16 * 8

    model.add(Convolution2D(16, 3, 3, border_mode='same',
                            init='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same',
                            init='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 8 * 8 * 16

    model.add(Convolution2D(16, 3, 3, border_mode='same',
                            init='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same',
                            init='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 4 * 4 * 32

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            init='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            init='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2 * 2 * 32

    model.add(Flatten())
    model.add(Dense(128, init='he_uniform', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def build_cnn_small():
    model = Sequential()

    # 64 * 64 * 1
    model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform', input_shape=(1, 64, 64)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 32 * 32 * 8

    model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform', input_shape=(1, 64, 64)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 16 * 16 * 8

    model.add(Convolution2D(16, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 8 * 8 * 16

    model.add(Convolution2D(16, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 4 * 4 * 32

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2 * 2 * 32

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, init='he_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def build_cnn_strange():
    model = Sequential()

    # 64 * 64 * 1
    model.add(Convolution2D(8, 7, 7, border_mode='same', activation='relu',
                            init='he_uniform',
                            subsample=(2, 2), input_shape=(1, 64, 64)))
    # 32 * 32 * 8

    model.add(Convolution2D(8, 5, 5, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 16 * 16 * 16

    model.add(Convolution2D(8, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 8 * 8 * 32

    model.add(Flatten())
    model.add(Dense(1024, init='he_uniform', activation='relu'))
    model.add(Dense(1024, init='he_uniform', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def main():
    six.print_('loading data')
    (train_x, train_y, val_x, val_y) = load_data()
    six.print_('load data complete')

    model = build_mlp()
    sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    six.print_('build model complete')

    six.print_('start training')
    model.fit(train_x, train_y, batch_size=128, nb_epoch=100,
              verbose=2,
              show_accuracy=True,
              shuffle=True,
              validation_data=(val_x, val_y))
    model.save_weights('weight.hdf5')


if __name__ == '__main__':
    main()
