#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("loading data ...")
train_data = np.loadtxt('./train_features.txt')
train_labels = np.loadtxt('./train_labels.txt')
validation_data = np.loadtxt('./validation_features.txt')
validation_labels = np.loadtxt('./validation_labels.txt')
test_data = np.loadtxt('./test_features.txt')
print("load data done.")

clf = SVC(
    C=1.0,
    kernel='rbf',
    shrinking=True,
    tol=1e-3,
    max_iter=20)

clf.fit(train_data, train_labels)
validation_pred = clf.predict(validation_data)
accuracy = accuracy_score(validation_labels, validation_pred)

print("accuracy = {}".format(accuracy))
