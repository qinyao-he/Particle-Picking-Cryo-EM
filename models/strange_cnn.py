from .common import *


def build():
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
