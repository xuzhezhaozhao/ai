#! /usr/bin/env python
# -*- coding=utf8 -*-


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import argparse
import numpy as np

FLAGS = None


def main():
    data = np.loadtxt(FLAGS.input_vec_file,
                      dtype=np.str,
                      delimiter=' ',
                      skiprows=2)
    D = data[:, :1].ravel()
    data = data[:, 1:-1].astype(np.float)
    print('data shape: {}'.format(data.shape))
    # normalize
    data = normalize(data, axis=1)

    y = KMeans(n_clusters=FLAGS.ncluster,
               max_iter=FLAGS.max_iter,
               tol=FLAGS.tol,
               precompute_distances=FLAGS.precompute_distances,
               n_jobs=FLAGS.njobs,
               verbose=1
               ).fit_predict(data)
    # y = DBSCAN(eps=0.1, metric='euclidean', n_jobs=-1).fit_predict(data)

    num = y.shape[0]
    clusters = dict()
    for i in range(num):
        label = y[i]
        if label not in clusters:
            clusters[label] = list()
        clusters[label].append(D[i])

    with open(FLAGS.output_cluster_file, 'w') as f:
        f.write(str(len(clusters)))
        f.write('\n')
        for label in clusters:
            f.write(str(len(clusters[label])))
            f.write(' ')
        f.write('\n')
        for label in clusters:
            f.write('__label__' + str(label) + ' ')
            for item in clusters[label]:
                f.write(item + ' ')
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_vec_file',
        type=str,
        required=True,
        help=''
    )

    parser.add_argument(
        '--ncluster',
        type=int,
        required=True,
        help=''
    )

    parser.add_argument(
        '--output_cluster_file',
        type=str,
        required=True,
        help=''
    )

    parser.add_argument(
        '--njobs',
        type=int,
        default=1,
        help=''
    )

    parser.add_argument(
        '--max_iter',
        type=int,
        default=300,
        help=''
    )
    parser.add_argument(
        '--tol',
        type=float,
        default=1e-4,
        help=''
    )
    parser.add_argument(
        '--precompute_distances',
        type=bool,
        default=True,
        help=''
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
