import numpy as np
import scipy.io as sio


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
