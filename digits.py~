import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

#Import data
train = pd.read_csv("train.csv", delimiter=',')
test = pd.read_csv("test.csv", delimiter=',')

#Get size of data
num_features = train.shape[1]-1
num_digits = train.shape[0]

#Get features and labels
features_train = train.columns[1:]
labels_train = train.columns[0]

features_test = test.columns[:]

#Create and train classifier
clf = KNeighborsClassifier(n_neighbors=5)

clf.fit(features_train, labels_train)
