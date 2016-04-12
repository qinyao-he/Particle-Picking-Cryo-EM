import os

import numpy as np
import scipy.io as sio


def main():
    MAT_DIR = './mat_new/train'
    train_x = np.zeros((0, 64, 64)).astype('uint8')
    train_y = np.zeros((0)).astype('uint8')
    for dirpath, dirnames, filenames in os.walk(MAT_DIR):
        print(dirpath)
        for filename in filenames:
            if filename == 'train.mat':
                data = sio.loadmat(os.path.join(dirpath, filename))
                train_x = np.append(train_x, data['train_x'], axis=0)
                train_y = np.append(train_y, data['train_y'].reshape(-1),
                                    axis=0)
    sio.savemat('data_new.mat', {
        'train_x': train_x,
        'train_y': train_y
    })


if __name__ == '__main__':
    main()
