from .common import *


def build():
    model = Sequential()

    model.add(Flatten(input_shape=(1, 64, 64)))

    model.add(Dense(1, activation='sigmoid'))

    return model
