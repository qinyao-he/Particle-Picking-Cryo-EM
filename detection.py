import os
import math

import numpy as np
import scipy
import scipy.io as sio
import skimage.transform
import matplotlib

import theano
import keras

from model import build_mlp


model = None


def init_model():
    global model
    model = build_mlp()
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.load_weights('weight-mlp-dropout.hdf5')


def prediction(img_patch):
    img_patch = np.reshape(img_patch, (1, 1, 64, 64))
    predict = model.predict(img_patch, verbose=0)
    return predict[0, 0]


def detection(img):
    PATCH_SIZE = 180
    (width, height) = img.shape
    stride = int(180 * 0.2)
    map_width = math.floor((width - PATCH_SIZE) / stride + 1)
    map_height = math.floor((height - PATCH_SIZE) / stride + 1)
    predict_map = np.zeros((map_width, map_height))
    result = []

    for i in range(0, map_width):
        for j in range(0, map_width):
            patch = img[i * stride: i * stride + PATCH_SIZE,
                        j * stride: j * stride + PATCH_SIZE]
            patch = skimage.transform.resize(patch, (64, 64))
            predict_map[i, j] = prediction(patch)

    for i in range(0, map_width):
        for j in range(0, map_height):
            di = [0, 1, 0, -1, -1, -1, 1, 1]
            dj = [1, 0, -1, 0, -1, 1, -1, 1]
            flag = True
            for k in range(8):
                ii = i + di[k]
                jj = j + dj[k]
                if (ii >= 0 and ii < map_width) \
                    and (jj >= 0 and jj < map_height) \
                        and (predict_map[i, j] < predict_map[ii, jj]):
                    flag = False
            if flag and predict_map[i, j] > 0.8:
                result.append((i * stride + PATCH_SIZE / 2,
                               j * stride + PATCH_SIZE / 2))
    return result


def main():
    # matplotlib.use('qt5agg')
    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Rectangle

    init_model()
    MAT_DIR = './mat/test'
    LABEL_DIR = './label/test'
    for dirpath, dirnames, filenames in os.walk(MAT_DIR):
        print(dirpath)
        for filename in filenames:
            if filename == 'full.mat':
                data = sio.loadmat(os.path.join(dirpath, filename))
                img = data['data']
                centers = detection(img)
                img_id = dirpath.split('/')[-1]
                label_file = os.path.join(LABEL_DIR, img_id + '.mat')
                labels = sio.loadmat(label_file)['label']
                distance = (lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2))

                TP = 0
                for x, y in labels:
                    for x_, y_ in centers:
                        if distance(x, y, x_, y_) < 36:
                            TP += 1
                            break
                precision = TP / len(centers)
                recall = TP / len(labels)
                f_score = 2 * (precision * recall) / (precision + recall)
                print(precision, recall, f_score)


if __name__ == '__main__':
    main()
