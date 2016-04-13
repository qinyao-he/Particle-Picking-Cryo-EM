import os

import numpy as np
import scipy.io as sio
import skimage.transform


def sliding(img, labels):
    patch_size = 180
    (width, height) = img.shape
    stride = 10
    map_width = int((width - patch_size) / stride + 1)
    map_height = int((height - patch_size) / stride + 1)

    distance = (lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2))

    print(len(labels))

    X = np.zeros((map_width, map_height, 64, 64)).astype('uint8')
    y = np.zeros((map_width, map_height)).astype('uint8')
    for i in range(0, map_width):
        for j in range(0, map_width):
            patch = img[i * stride: i * stride + patch_size,
                        j * stride: j * stride + patch_size]
            X[i, j] = skimage.transform.resize(patch, (64, 64))
            x_center = j * stride + patch_size / 2
            y_center = i * stride + patch_size / 2
            dist = distance(labels[:, 0], labels[:, 1], x_center, y_center)
            cond = np.where(dist <= 36)[0]
            if (len(cond) == 0):
                y[i, j] = 0
            else:
                y[i, j] = 1

    X = X.reshape(-1, 64, 64)
    y = y.reshape(-1)
    X = X[y == 0]
    y = y[y == 0]

    X_extend = np.zeros((len(labels) * 3 * 3 * 8, 64, 64)).astype('uint8')
    y_extend = np.zeros((len(labels) * 3 * 3 * 8)).astype('uint8')
    cnt = 0
    for i in range(len(labels)):
        x_ = labels[i, 0]
        y_ = labels[i, 1]
        for i_offset in range(-4, 5, 4):
            for j_offset in range(-4, 5, 4):
                x1 = x_ + j_offset - patch_size / 2
                x2 = x_ + j_offset + patch_size / 2
                y1 = y_ + i_offset - patch_size / 2
                y2 = y_ + i_offset + patch_size / 2
                if (x1 >= 0 and x2 < height) and (y1 >= 0 and y2 < width):
                    patch = img[y1: y2, x1: x2]
                    patch = skimage.transform.resize(patch, (64, 64))
                    X_extend[cnt] = patch
                    X_extend[cnt + 1] = skimage.transform.rotate(patch, 90)
                    X_extend[cnt + 2] = skimage.transform.rotate(patch, 180)
                    X_extend[cnt + 3] = skimage.transform.rotate(patch, 270)
                    patch = np.transpose(patch)
                    X_extend[cnt + 4] = patch
                    X_extend[cnt + 5] = skimage.transform.rotate(patch, 90)
                    X_extend[cnt + 6] = skimage.transform.rotate(patch, 180)
                    X_extend[cnt + 7] = skimage.transform.rotate(patch, 270)
                    for k in range(8):
                        y_extend[cnt + k] = 1
                cnt += 8

    X_extend = X_extend[y_extend == 1]
    y_extend = y_extend[y_extend == 1]

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    X = np.concatenate([X[::len(y_extend)], X_extend], axis=0).astype('uint8')
    y = np.concatenate([y[:len(y_extend)], y_extend], axis=0).astype('uint8')
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
