from .common import *


def build():
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
