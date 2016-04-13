import math
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def judge(pd_centers, gt_centers):
    TP = 0
    for x in gt_centers:
        for x_ in pd_centers:
            if distance(x[0], x[1], x_[0], x_[1]) < 36:
                TP += 1
                break
    precision = TP / len(pd_centers)
    recall = TP / len(gt_centers)
    f_score = 2 * (precision * recall) / (precision + recall)
    print(len(pd_centers), len(gt_centers), TP, precision, recall, f_score)


def get_list(filename):
    f = open(filename)
    lines = f.readlines()
    res = []
    for l in lines:
        part = (l.strip()).split(' ')
        res.append([int(float(part[0])), int(float(part[1]))])
    return res


def save_list(filename, out_list):
    fout = open(filename, 'w')
    for x, y in out_list:
        fout.write(str(int(y)) + " " + str(int(x)) + '\n')
    fout.close()


def main():
    centers = get_list('out_center.txt')
    labels = get_list('142-label.txt')
    judge(centers, labels)
    n_class = int(len(centers) * 0.18)
    est = KMeans(n_clusters=n_class, max_iter=1000)
    est.fit(centers)
    new_list = []
    for x, y in est.cluster_centers_:
        min_num = 10000
        min_x = -1
        min_y = -1
        for x_, y_ in centers:
            dist = distance(x, y, x_, y_)
            if (dist < min_num) or (min_x == -1):
                min_num = dist
                min_x = x_
                min_y = y_
        new_list.append([min_x, min_y])
    judge(new_list, labels)
    judge(est.cluster_centers_, labels)

    # db = DBSCAN(eps=0.3, min_samples=180).fit(centers)
    # print(db.core_sample_indices_)
    # judge(new_list, labels)
    # print(est.cluster_centers_)
    # save_list('result.txt', est.cluster_centers_)
    # af = AffinityPropagation(preference=180).fit(centers)
    # judge(af.cluster_centers_, labels)

if __name__ == '__main__':
    main()
