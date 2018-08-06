# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# Which features are available in the dataset?
# Which features are categorical?
# Which features are numerical?
# Which features are mixed data types?
# Which features may contain errors or typos?
# Which features contain blank, null or empty values?
# What are the data types for various features?
train_df.columns.values
train_df.head()
train_df.tail()
train_df.info()
test_df.info()
train_df.dtypes

# What is the distribution of numerical feature values across the samples?
# What is the distribution of categorical features?
len(train_df[train_df.Survived == 1])
len(train_df[train_df.Age > 50])
len(train_df[train_df.SibSp == 0])
len(train_df[train_df.Parch == 0])
len(train_df[train_df.Fare > 100])

train_df.describe()
train_df.describe(include='all')
train_df.describe(include=['O'])
train_df.describe(percentiles=[.61, .62])

# pivoting features analysis...
train_df[['Pclass', 'Survived']] \
    .groupby(['Pclass'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=False)

train_df[["Sex", "Survived"]] \
    .groupby(['Sex'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=False)

train_df[["SibSp", "Survived"]] \
    .groupby(['SibSp'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=False)

train_df[["Parch", "Survived"]] \
    .groupby(['Parch'], as_index=False) \
    .mean() \
    .sort_values(by='Survived', ascending=False)
