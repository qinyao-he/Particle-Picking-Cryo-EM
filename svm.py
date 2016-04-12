import os
import six

from load import load_data

from sklearn.svm import SVC


def main():
    six.print_('loading data')
    train_x, train_y, val_x, val_y = load_data()
    train_x = train_x.reshape(-1, 64 * 64)
    val_x = val_x.reshape(-1, 64 * 64)
    six.print_('load data complete')

    clf = SVC(C=0.1, kernel='linear', verbose=True, max_iter=1000)
    six.print_('start training')
    clf.fit(train_x, train_y)

    acc = sum(val_y == clf.predict(val_x)) / len(val_y)
    print(acc)


if __name__ == '__main__':
    main()
