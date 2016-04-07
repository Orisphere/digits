import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

#Import data
train = pd.read_csv("train.csv", delimiter=',')
test = pd.read_csv("test.csv", delimiter=',')

#Get size of data
num_features = train.shape[1]-1
num_digits = train.shape[0]

#Get features and labels
features_cols = train.columns[1:]
labels_cols = train.columns[0]

features_train = train[features_cols]
labels_train = train[labels_cols]


features_test = test

#Create cross-validation set
train_X, test_X, train_y, test_y = cross_validation.train_test_split(features_train, labels_train, test_size = 0.2, random_state=0)

#Create and train classifier
clf = GaussianNB()

clf.fit(train_X, train_y)

#Get f1 score
pred_train = clf.predict(train_X)
pred_test = clf.predict(test_X)

#f1_train = f1_score(train_y, pred_train)
f1_test = f1_score(test_y, pred_test)

#print('F1 score on training data is: ' + str(f1_train))
print('F1 score on testing data is: ' + str(f1_test))

