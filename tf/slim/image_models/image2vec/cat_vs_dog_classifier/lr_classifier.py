#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

print("loading data ...")
train_data = np.loadtxt('./train_features.txt')
train_labels = np.loadtxt('./train_labels.txt')
validation_data = np.loadtxt('./validation_features.txt')
validation_labels = np.loadtxt('./validation_labels.txt')

# concatenate train and validation data, use cross validation
# train_data = np.concatenate((train_data, validation_data), axis=0)
# train_labels = np.concatenate((train_labels, validation_labels), axis=0)

test_data = np.loadtxt('./test_features.txt')
print("load data done.")

clf = LogisticRegression(
    random_state=None,
    penalty='l2',
    tol=1e-6,
    C=3.0,
    solver='liblinear',
    max_iter=40)

# clf = LogisticRegressionCV(
    # Cs=10,
    # cv=10,
    # penalty='l2',
    # scoring='accuracy',
    # solver='liblinear',
    # tol=1e-6,
    # max_iter=40,
    # refit=True,
    # n_jobs=1,
    # verbose=1)

print("fitting ...")
clf.fit(train_data, train_labels)
print("fitting done.")

train_pred = clf.predict(train_data)
train_prob = clf.predict_proba(train_data)
accuracy = accuracy_score(train_labels, train_pred)
loss = log_loss(train_labels, train_prob)
print("[train] accuracy = {}, loss = {}".format(accuracy, loss))

validation_pred = clf.predict(validation_data)
validation_prob = clf.predict_proba(validation_data)
accuracy = accuracy_score(validation_labels, validation_pred)
loss = log_loss(validation_labels, validation_prob)
print("[validation] accuracy = {}, loss = {}".format(accuracy, loss))

test_prob = clf.predict_proba(test_data)
with open('test_prob.txt', 'w') as fout, \
        open('test.txt', 'r') as fin:
    for prob in test_prob:
        src = fin.readline().strip()
        fout.write(src + ' ')
        fout.write(str(prob[1]) + '\n')
