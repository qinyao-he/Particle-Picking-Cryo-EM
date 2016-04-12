from .common import *


def build():
    model = Sequential()

    # 64 * 64 * 1
    model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu',
                            init='he_uniform', input_shape=(1, 64, 64)))
    model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 32 * 32 * 32

    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 16 * 16 * 32

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 8 * 8 * 64

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 4 * 4 * 64

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2 * 2 * 128

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, init='he_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model
