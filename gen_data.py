import os

import numpy as np
import scipy.io as sio
import skimage.transform


def sliding(img, labels):
    patch_size = 180
    (width, height) = img.shape
    stride = 36
    map_width = int((width - patch_size) / stride + 1)
    map_height = int((height - patch_size) / stride + 1)

    distance = (lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2))

    print(len(labels))

    X_negative = np.zeros((map_width, map_height, 64, 64)).astype('uint8')
    y_negative = np.zeros((map_width, map_height)).astype('uint8')
    for i in range(0, map_width):
        for j in range(0, map_width):
            patch = img[i * stride: i * stride + patch_size,
                        j * stride: j * stride + patch_size] / 256.0
            X_negative[i, j] = skimage.transform.resize(patch, (64, 64)) * 256
            x_center = j * stride + patch_size / 2
            y_center = i * stride + patch_size / 2
            dist = distance(labels[:, 0], labels[:, 1], x_center, y_center)
            cond = np.where(dist <= 36)[0]
            if (len(cond) == 0):
                y_negative[i, j] = 0
            else:
                y_negative[i, j] = 1

    X_negative = X_negative.reshape(-1, 64, 64)
    y_negative = y_negative.reshape(-1)
    X_negative = X_negative[y_negative == 0]
    y_negative = y_negative[y_negative == 0]

    X_positive = np.zeros((len(labels) * 9 * 9, 64, 64)).astype('uint8')
    y_positive = np.zeros((len(labels) * 9 * 9)).astype('uint8')
    cnt = 0
    for i in range(len(labels)):
        x = labels[i, 0]
        y = labels[i, 1]
        for i_offset in range(-4, 5, 4):
            for j_offset in range(-4, 5, 4):
                x1 = x + j_offset - patch_size / 2
                x2 = x + j_offset + patch_size / 2
                y1 = y + i_offset - patch_size / 2
                y2 = y + i_offset + patch_size / 2
                if (x1 >= 0 and x2 <= height) and (y1 >= 0 and y2 <= width):
                    patch = img[y1: y2, x1: x2] / 256.0
                    patch = skimage.transform.resize(patch, (64, 64)) * 256
                    X_positive[cnt] = patch
                    y_positive[cnt] = 1
                cnt += 1

    X_positive = X_positive[y_positive == 1]
    y_positive = y_positive[y_positive == 1]

    indices = np.arange(len(X_negative))
    np.random.shuffle(indices)
    X_negative = X_negative[indices]
    y_negative = y_negative[indices]

    X = np.concatenate([X_negative[:len(y_positive)], X_positive], axis=0) \
        .astype('uint8')
    y = np.concatenate([y_negative[:len(y_positive)], y_positive], axis=0) \
        .astype('uint8')
    print(sum(y == 0), sum(y == 1))
    sio.savemat('data_new.mat', {
        'train_x': X,
        'train_y': y
    })
    return (X, y)


def main():
    MAT_PATH = './mat/train'
    LABEL_PATH = './label/train'

    for dirpath, dirnames, filenames in os.walk(MAT_PATH):
        print(dirpath)
        for filename in filenames:
            if filename == 'full.mat':
                img = sio.loadmat(os.path.join(dirpath, filename))['data']
                img_id = dirpath.split('/')[-1]
                label_file = os.path.join(LABEL_PATH, img_id + '.mat')
                labels = sio.loadmat(label_file)['label']
                sliding(img, labels)


if __name__ == '__main__':
    main()
