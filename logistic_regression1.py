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



os.chdir('C:\\Users\\yanran.zhou\\machine-learning-ex2\\ex2')

df= pd.read_csv('ex2data1.txt',header=None,index_col=False)


print(df.shape)
print(df.describe())
print(df.dtypes)

admitted = df.loc[df[2]==1]
notadmitted = df.loc[df[2]==0]

#plt.scatter(admitted[0],admitted[1],c='yellow')
#plt.scatter(notadmitted[0],notadmitted[1],c='r')
#plt.show()

predictors = df[[0,1]].values
#predictor2 = df[1].values
target = df[2].values


classifier = OneVsRestClassifier(LogisticRegression(penalty='l1')).fit(predictors, target)


#print(model.predict_proba(pred_x_test))
print('Coefficients: \n', classifier.coef_)
print('intercept: \n', classifier.intercept_)
print (classifier.score(predictors, target))

coef = classifier.coef_
intercept = classifier.intercept_



ex1 = np.linspace(30, 100, 100)
ex2 = -(coef[:, 0] * ex1 + intercept[:, 0]) / coef[:,1]

plt.scatter(admitted[0],admitted[1],c='yellow')
plt.scatter(notadmitted[0],notadmitted[1],c='r')
plt.plot(ex1, ex2, color='r', label='decision boundary');
plt.show()