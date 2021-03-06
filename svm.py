import os
import six
import pickle

from sklearn.svm import SVC
from sklearn import decomposition

from load import load_data


def main():
    six.print_('loading data')
    train_x, train_y, val_x, val_y = load_data()
    train_x = train_x.reshape(-1, 64 * 64)
    val_x = val_x.reshape(-1, 64 * 64)
    six.print_('load data complete')

    six.print_('start PCA')
    try:
        pca = pickle.load(open('pca.pickle', 'rb'))
    except:
        pca = decomposition.PCA(n_components=8*8)
        pca.fit(train_x[:])
    train_x = pca.transform(train_x)
    six.print_('PCA complete')

    clf = SVC(C=0.0001, kernel='linear', verbose=True, max_iter=100)
    six.print_('start training')
    clf.fit(train_x, train_y)
    six.print_('training complete')

    val_x = pca.transform(val_x)
    acc = sum(val_y == clf.predict(val_x)) / float(len(val_y))
    print(acc)

    pickle.dump(pca, open('pca.pickle', 'wb'))
    pickle.dump(clf, open('svm.pickle', 'wb'))


if __name__ == '__main__':
    main()
