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

os.chdir('C:\\Users\\yanran.zhou\\machine-learning-ex2\\ex2')

df= pd.read_csv('ex2data2.txt',header=None,index_col=False)


print(df.shape)
print(df.describe())
print(df.dtypes)
print(df.head)

accepted = df.loc[df[2]==1]
rejected = df.loc[df[2]==0]

#plt.scatter(accepted[0],accepted[1],c='r',marker='+')
#plt.scatter(rejected[0],rejected[1],c='b',marker='o')
#plt.show()

def map_features(f1, f2, order=1):
    '''map the f1 and f2 to its higher order polynomial'''
    assert order >= 1
    def iter():
        for i in range(1, order + 1):
            for j in range(i + 1):
                yield np.power(f1, i - j) * np.power(f2, j)
    return np.vstack(iter())

out = map_features(df[0], df[1], order=6)
X = out.transpose()
Y = df[2]    


classifier = OneVsRestClassifier(LogisticRegression(penalty='l1', C=1)).fit(X, Y)
print ('Coefficents: ', classifier.coef_)
print ('Intercept: ', classifier.intercept_)
print ('Accuracy: ', classifier.score(X, Y))


def draw_boundary(classifier):
    dim = np.linspace(-1, 1.5, 1000)
    dx, dy = np.meshgrid(dim, dim)
    v = map_features(dx.flatten(), dy.flatten(), order=6)
    z = (np.dot(classifier.coef_, v) + classifier.intercept_).reshape(1000, 1000)
    CS = plt.contour(dx, dy, z, levels=[0], colors=['r'])    

plt.scatter(accepted[0],accepted[1],c='r',marker='+')
plt.scatter(rejected[0],rejected[1],c='b',marker='o')
draw_boundary(classifier)
plt.legend()
plt.show()

##### regularization #####

overfitter = OneVsRestClassifier(LogisticRegression(penalty='l1', C=1000)).fit(X, Y)
plt.scatter(accepted[0],accepted[1],c='r',marker='+')
plt.scatter(rejected[0],rejected[1],c='b',marker='o')
draw_boundary(overfitter)
plt.legend()
plt.show()

##### underfitting ##########

underfitter = OneVsRestClassifier(LogisticRegression(penalty='l1', C=0.01)).fit(X, Y)
plt.scatter(accepted[0],accepted[1],c='r',marker='+')
plt.scatter(rejected[0],rejected[1],c='b',marker='o')
draw_boundary(underfitter)
plt.legend()
plt.show()

