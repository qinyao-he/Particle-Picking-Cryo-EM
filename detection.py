import os
import six
import pickle

import numpy as np
import scipy.io as sio
import skimage.transform
import matplotlib

model = None
svm = None
pca = None


def init_model():
    global model
    import models.mlp
    model = models.mlp.build()
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.load_weights('weight-mlp.hdf5')


def init_svm():
    global svm
    global pca
    svm = pickle.load(open('svm.pickle', 'rb'))
    pca = pickle.load(open('pca.pickle', 'rb'))


def prediction(img_batch):
    img_batch = np.reshape(img_batch, (-1, 1, 64, 64))
    predict = model.predict(img_batch, verbose=0)
    return predict


def prediction_svm(img_batch):
    img_batch = np.reshape(img_batch, (-1, 64 * 64))
    img_batch = pca.transform(img_batch)
    predict = svm.predict(img_batch)
    return predict


def non_max_suppression(centers, threshold):
    if len(centers) == 0:
        return []

    pick = []
    x1 = centers[:, 0] - 90
    y1 = centers[:, 1] - 90
    x2 = centers[:, 0] + 90
    y2 = centers[:, 1] + 90
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = w * h / float(180 * 180)

        idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > threshold)[0])))

    return centers[pick].astype("int")


def detection(img):
    PATCH_SIZE = 180
    (width, height) = img.shape
    stride = 10
    map_width = int((width - PATCH_SIZE) / stride + 1)
    map_height = int((height - PATCH_SIZE) / stride + 1)

    img_batch = np.zeros((map_width, map_height, 64, 64))
    for i in range(0, map_width):
        for j in range(0, map_width):
            patch = img[i * stride: i * stride + PATCH_SIZE,
                        j * stride: j * stride + PATCH_SIZE] / 256.0
            img_batch[i, j] = skimage.transform.resize(patch, (64, 64))
    predict_map = prediction(img_batch.reshape(-1, 64, 64))
    predict_map = predict_map.reshape((map_width, map_height))

    result = []
    for i in range(0, map_width):
        for j in range(0, map_width):
            if predict_map[i, j] > 0.95:
                result.append((i * stride + PATCH_SIZE / 2,
                               j * stride + PATCH_SIZE / 2))
    # return non_max_suppression(np.array(result), 0.3)
    return result


def main():
    # matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    init_svm()
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
                precision = float(TP) / len(centers)
                recall = float(TP) / len(labels)
                f_score = 2 * (precision * recall) / (precision + recall)
                six.print_(precision, recall, f_score)

                # f = open(dirpath.split('/')[-1] + '-predict.txt', 'w')
                # for x, y in centers:
                #    f.write(str(x) + ' ' + str(y) + '\n')
                # f.close()
                # f = open(dirpath.split('/')[-1] + '-label.txt', 'w')
                # for x, y in labels:
                #     f.write(str(x) + ' ' + str(y) + '\n')
                # f.close()

                img = img / np.float32(256)
                plt.imshow(img, cmap=plt.cm.gray)
                currentAxis = plt.gca()
                for x, y in labels:
                    currentAxis.add_patch(Rectangle((y - 90, x - 90),
                                                    180, 180, fill=None,
                                                    alpha=1))
                for x, y in centers:
                    currentAxis.add_patch(Rectangle((y - 90, x - 90),
                                                    180, 180, fill=None,
                                                    alpha=1, color='blue'))
                plt.show()


if __name__ == '__main__':
    main()
