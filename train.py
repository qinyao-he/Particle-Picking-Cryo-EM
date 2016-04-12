import os
import six
import argparse
import importlib

import theano
import keras

from keras.optimizers import SGD

from load import load_data


def main():
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('--model', type=str)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--output', type=str, default='weight')
    args = parser.parse_args()

    model = importlib.import_module(args.model).build()

    six.print_('loading data')
    (train_x, train_y, val_x, val_y) = load_data()
    six.print_('load data complete')

    sgd = SGD(lr=args.lr,
              decay=args.decay,
              momentum=args.momentum,
              nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    six.print_('build model complete')

    six.print_('start training')
    model.fit(train_x, train_y, batch_size=args.batch, nb_epoch=args.epoch,
              verbose=2,
              show_accuracy=True,
              shuffle=True,
              validation_data=(val_x, val_y))
    model.save_weights(args.output + '.hdf5')


if __name__ == '__main__':
    main()
