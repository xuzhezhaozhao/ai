#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

train_data = np.loadtxt('./train_features.txt')
train_labels = np.loadtxt('./train_labels.txt')
validation_data = np.loadtxt('./validation_features.txt')
validation_labels = np.loadtxt('./validation_labels.txt')
test_data = np.loadtxt('./test_features.txt')

clf = LogisticRegression(
    random_state=None,
    penalty='l2',
    tol=1e-6,
    C=3.0,
    solver='liblinear',
    max_iter=10)

clf.fit(train_data, train_labels)
validation_pred = clf.predict(validation_data)
validation_prob = clf.predict_proba(validation_data)

accuracy = accuracy_score(validation_labels, validation_pred)
loss = log_loss(validation_labels, validation_prob)

print("accuracy = {}, loss = {}".format(accuracy, loss))
