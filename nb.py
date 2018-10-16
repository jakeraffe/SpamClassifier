# -*- coding: utf-8 -*-
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

dataset = pd.read_csv('spambase.data').as_matrix()
np.random.shuffle(dataset) # Inplace shuffle of the data

X = dataset[:, :48]
Y = dataset[:,-1]

Xtrain = X[:-100,] # First 100 rows
Ytrain = Y[:-100,]
Xtest = X[-100:,] # Last 100 rows
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain,Ytrain)

print("Classification rate for Naive Bayes", model.score(Xtest,Ytest))

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain,Ytrain)
print("Classification rate for AdaBoostClassifier", model.score(Xtest,Ytest))
