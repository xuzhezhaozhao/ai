import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as sk_auc


def auc(labels, preds, n_bins=100):
    postive_cnt = sum(labels)
    negative_cnt = len(labels) - postive_cnt
    total_case = postive_cnt * negative_cnt
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i]/bin_width)
        if labels[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i]*accumulated_neg +
                           pos_histogram[i]*neg_histogram[i]*0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)


if __name__ == '__main__':

    y = np.array([1, 0, 0, 0, 1, 0, 1, 0])
    pred = np.array([0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7])

    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    print("sklearn:", sk_auc(fpr, tpr))
    print("reimplementation:", auc(y, pred))
