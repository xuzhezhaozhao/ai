#! /usr/bin/env python
# -*- coding=utf8 -*-


from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import argparse
import numpy as np
import time

FLAGS = None


def main():
    video_data = np.loadtxt(FLAGS.input_video_vec_file,
                            dtype=np.str,
                            delimiter=' ',
                            skiprows=2)
    video_dict = video_data[:, :1].ravel()
    video_data = video_data[:, 1:].astype(np.float)
    video_data = normalize(video_data, axis=1)
    print(video_data[0])
    print('video_data shape: {}'.format(video_data.shape))

    all_data = np.loadtxt(FLAGS.input_all_vec_file,
                          dtype=np.str,
                          delimiter=' ',
                          skiprows=2)
    all_dict = all_data[:, :1].ravel()
    all_data = all_data[:, 1:-1].astype(np.float)
    all_data = normalize(all_data, axis=1)
    print('all_data shape: {}'.format(all_data.shape))

    kmeans = KMeans(n_clusters=FLAGS.ncluster,
                    max_iter=FLAGS.max_iter,
                    tol=FLAGS.tol,
                    precompute_distances=FLAGS.precompute_distances,
                    n_jobs=FLAGS.njobs,
                    verbose=1)
    kmeans.fit(video_data)
    all_labels = kmeans.predict(all_data)

    clusters = dict()
    for idx, label in enumerate(all_labels):
        if label not in clusters:
            clusters[label] = list()
        clusters[label].append(all_dict[idx])
    with open(FLAGS.output_cluster_file, 'w') as f:
        f.write(str(len(clusters)) + '\n')
        for label in clusters:
            f.write(str(len(clusters[label])))
            f.write(' ')
        f.write('\n')
        fomat_time = time.strftime("%Y%m%d%H", time.localtime())
        for label in clusters:
            f.write('__label__' + fomat_time + '__' + str(label) + ' ')
            for item in clusters[label]:
                f.write(item + ' ')
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_video_vec_file',
        type=str,
        required=True,
        help=''
    )

    parser.add_argument(
        '--input_all_vec_file',
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
