
import numpy  as np 
from pandas import Series, DataFrame
import pandas as pd
import os
import matplotlib.pylab as plt
from sklearn import datasets, linear_model
#from sklearn.cross_validation import train_test_split


os.chdir('C:\\Users\\yanran.zhou\\machine-learning-ex1\\ex1')

df = pd.read_csv('ex1data1.txt',header=None, index_col=False)

print(df.dtypes)
print(df.describe())
length=(len(df))

#plt.scatter(df[0],df[1])
#plt.show()


predictor = df[0].values
target = df[1].values

predictor = predictor.reshape(length,1)
target = target.reshape(length,1)

pred_x_train = predictor[:-40]
pred_y_train = target[:-40]

pred_x_test = predictor[-40:]
pred_y_test = target[-40:]

print(pred_x_train.shape)
print(pred_y_train.shape)
print(pred_x_test.shape)
print(pred_y_test.shape)


regr = linear_model.LinearRegression()
regr.fit(pred_x_train, pred_y_train)

print('Coefficients: \n', regr.coef_)

print("Mean squared error: %.2f"
      % np.mean((regr.predict(pred_x_test) - pred_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(pred_x_test, pred_y_test))


plt.scatter(pred_x_test, pred_y_test,  color='black')
plt.plot(pred_x_test, regr.predict(pred_x_test), color='blue',
         linewidth=3)
plt.title('Linear regression')
plt.show()


plt.scatter(regr.predict(pred_x_train),regr.predict(pred_x_train)-pred_y_train, c = 'b', s = 40, alpha=0.5)
plt.scatter(regr.predict(pred_x_test),regr.predict(pred_x_test)-pred_y_test,c='g',s=40)
plt.hlines(y=0,xmin=0,xmax=30)
plt.title('residual plot for test and train data')
plt.ylabel('residuals')
plt.show()
