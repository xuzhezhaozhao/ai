#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_data = np.loadtxt('./train_features.txt')
train_labels = np.loadtxt('./train_labels.txt')
validation_data = np.loadtxt('./validation_features.txt')
validation_labels = np.loadtxt('./validation_labels.txt')
test_data = np.loadtxt('./test_features.txt')

clf = SVC(
    C=1.0,
    kernel='rbf',
    shrinking=True,
    tol=1e-3,
    max_iter=-1)

clf.fit(train_data, train_labels)
validation_pred = clf.predict(validation_data)
accuracy = accuracy_score(validation_labels, validation_pred)

print("accuracy = {}".format(accuracy))
