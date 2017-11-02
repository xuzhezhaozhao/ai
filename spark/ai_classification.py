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
            uin = str(s[1])
            title = s[4]
            content = s[5]
            try:
                lable = int(s[6])
            except:
                print("line: ", cnt)
            else:
                df.append( (lable, [uin] + tokenize(title) + tokenize(content), ) )
            cnt += 1

    return df


if __name__ == "__main__":
    spark = SparkSession \
    .builder \
    .appName("Python Spark text classification example") \
    .config("spark.executor.memory", "6g") \
    .getOrCreate()

    # df = spark.createDataFrame( extract("post_tagged.csv"), ["label", "words"])
    f = open('words.pickle')
    words = pickle.load(f)
    df = spark.createDataFrame(words, ["label", "words"])


    # hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    featurizedData = hashingTF.transform(df)
    # alternatively, CountVectorizer can also be used to get term frequency vectors
    df = None

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    featurizedData = None

    # normalizer
    normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=2.0)
    l2Normdata = normalizer.transform(rescaledData)
    rescaledData = None


    with open("preprocessed.svm", "w") as f:
        c = l2Normdata.collect()
        for row in c:
            d = row.asDict()
            normFeatures = d['normFeatures']
            label = d['label']

            indices = normFeatures.indices.tolist()
            values = normFeatures.values.tolist()
            f.write(str(label))
            for pair in zip(indices, values):
                f.write(' ' + str(pair[0]) + ':' + str(pair[1]))
            f.write('\n')
            f.flush()

    
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
