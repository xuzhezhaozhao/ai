#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhezhaoxu
# Created Time : Mon 23 Oct 2017 02:44:55 PM CST
# File Name: ai_classification.py
# Description:
"""

# http://spark.apache.org/docs/latest/ml-features.html#feature-extractors

import jieba

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Normalizer

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import pickle

def load_stop_words():
    words = open("stop_words.csv").read().split('\n')
    uwords = []
    for word in words:
        uwords.append(unicode(word, 'utf-8'))
    return uwords

# tokenize a string
def tokenize(str):
    tokens = jieba.cut(str)
    utokens = []
    stop_words = load_stop_words()
    for token in tokens:
        if token in stop_words:
            continue
        utokens.append(token)
    return utokens

 
def extract(filename):
    df = []

    with open(filename, "rU") as f:
        cnt = 0
        for line in f:
            s = line.split('\t')
            title = s[4]
            content = s[5]
            try:
                lable = int(s[6])
            except:
                print("line: ", cnt)
            else:
                df.append( (lable, tokenize(title) + tokenize(content), ) )
            cnt += 1

    return df


if __name__ == "__main__":
    spark = SparkSession \
    .builder \
    .appName("Python Spark text classification example") \
    .config("spark.executor.memory", "6g") \
    .getOrCreate()
    
    # Load training data
    data = spark.read.format("libsvm").load("preprocessed.svm")

    # Split the data into train and test
    splits = data.randomSplit([0.6, 0.4], 1234)
    train = splits[0]
    test = splits[1]

    # create the trainer and set its parameters
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

    # train the model
    model = nb.fit(train)

    # select example rows to display.
    predictions = model.transform(test)
    predictions.show()

    # compute accuracy on the test set
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test set accuracy = " + str(accuracy))

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="f1")
    f1 = evaluator.evaluate(predictions)
    print("Test set f1 = " + str(f1))

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="weightedPrecision")
    precision = evaluator.evaluate(predictions)
    print("Test set precision = " + str(precision))

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="weightedRecall")
    recall = evaluator.evaluate(predictions)
    print("Test set recall = " + str(recall))


