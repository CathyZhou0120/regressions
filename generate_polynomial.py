import numpy  as np 
from pandas import Series, DataFrame
import pandas as pd
import os
import matplotlib.pylab as plt
import math
import random
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

os.chdir('C:\\Users\\yanran.zhou\\machine-learning-ex2\\ex2')

df= pd.read_csv('ex2data2.txt',header=None,index_col=False)


accepted = df.loc[df[2]==1]
rejected = df.loc[df[2]==0]

X = np.arange(6).reshape(3, 2)
print(type(X))
print(X.shape)

predictors = df[[0,1]]

predictors=predictors.values

poly = PolynomialFeatures(degree=3)
predictors_ = (poly.fit_transform(predictors))

classifier = OneVsRestClassifier(LogisticRegression(penalty='l1', C=1)).fit(predictors_, df[2])
print ('Coefficents: ', classifier.coef_)
print ('Intercept: ', classifier.intercept_)
print ('Accuracy: ', classifier.score(predictors_, df[2]))



