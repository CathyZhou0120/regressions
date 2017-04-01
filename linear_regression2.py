import numpy  as np 
from pandas import Series, DataFrame
import pandas as pd
import os
import matplotlib.pylab as plt
import math
import random

def covariance( x, y): 
	n = len( x) 
	return dot( de_mean( x), de_mean( y)) / (n - 1)


def correlation( x, y): 
	stdev_x = standard_deviation( x) 
	stdev_y = standard_deviation( y) 
	if stdev_x > 0 and stdev_y > 0: 
		return covariance( x, y) / stdev_x / stdev_y 
	else: 
		return 0

def sum_of_squares( v): 
	""" v_1 * v_1 + ... + v_n * v_n""" 
	return dot( v, v)
def dot( v, w): 
	""" v_1 * w_1 + ... + v_n * w_n""" 
	return sum( v_i * w_i for v_i, w_i in zip( v, w))


def mean( x): 
	return sum( x) / len( x)

def de_mean( x): 
	""" translate x by subtracting its mean (so the result has mean 0)""" 
	x_bar = mean( x) 
	return [x_i - x_bar for x_i in x]

def standard_deviation( x): 
	return math.sqrt( variance( x))

def variance( x): 
	""" assumes x has at least two elements""" 
	n = len( x) 
	deviations = de_mean( x) 
	return sum_of_squares( deviations) / (n - 1)



def predict( alpha, beta, x_i): 
	return beta * x_i + alpha

def error( alpha, beta, x_i, y_i): 
	""" the error from predicting beta * x_i + alpha when the actual value is y_i""" 
	return y_i - predict( alpha, beta, x_i)

def sum_of_squared_errors( alpha, beta, x, y): 
	return sum( error( alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip( x, y))

def least_squares_fit( x, y): 
	""" given training values for x and y, find the least-squares values of alpha and beta""" 
	beta = correlation( x, y) * standard_deviation( y) / standard_deviation( x) 
	alpha = mean( y) - beta * mean( x) 
	return alpha, beta

def total_sum_of_squares( y): 
	""" the total squared variation of y_i's from their mean""" 
	return sum( v ** 2 for v in de_mean( y)) 

def r_squared( alpha, beta, x, y): 
	""" the fraction of variation in y captured by the model, which equals 1 - the fraction of variation in y not captured by the model""" 
	return 1.0 - (sum_of_squared_errors( alpha, beta, x, y) /total_sum_of_squares( y)) 


def minimize_stochastic( target_fn, gradient_fn, x, y, theta_0, alpha_0 = 0.01):
    data = zip( x, y) 
    theta = theta_0 
    # initial guess 
    alpha = alpha_0 
    # initial step size 
    min_theta, min_value = None, float(" inf")
     # the minimum so far 
    iterations_with_no_improvement = 0 
     # if we ever go 100 iterations with no improvement, stop 
    while iterations_with_no_improvement < 100: 
        value = sum( target_fn( x_i, y_i, theta) for x_i, y_i in data ) 
        if value < min_value: 
        # if we've found a new minimum, remember it # and go back to the original step size 
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0 
            alpha = alpha_0 
        else: 
        # otherwise we're not improving, so try shrinking the step size 
            iterations_with_no_improvement += 1 
            alpha *= 0.9 # and take a gradient step for each of the data points 
        for x_i, y_i in in_random_order(data): 
        	gradient_i = gradient_fn( x_i, y_i, theta) 
        	theta = vector_subtract( theta, scalar_multiply( alpha, gradient_i)) 
    return min_theta

def in_random_order( data): 
    """ generator that returns the elements of data in random order""" 
    indexes = [i for i, _ in enumerate( data)] 
    # create a list of indexes 
    random.shuffle( indexes) # shuffle them 
    for i in indexes: # return the data in that order 
        yield data[ i]



os.chdir('C:\\Users\\yanran.zhou\\machine-learning-ex1\\ex1')

df = pd.read_csv('ex1data1.txt',header=None, index_col=False)

print(df.dtypes)
print(df.describe())
length=(len(df))


predictor = df[0].values
target = df[1].values

predictor = predictor.reshape(length,1)
target = target.reshape(length,1)

alpha, beta = least_squares_fit( predictor, target)
print(r_squared( alpha, beta, predictor, target))

plt.scatter(predictor, target,  color='black')
plt.plot(predictor, predictor*beta+alpha, color='blue',
         linewidth=3)

plt.title('Linear regression')
plt.show()

####### gradient descent ##############

def squared_error( x_i, y_i, theta): 
	alpha, beta = theta 
	return error( alpha, beta, x_i, y_i) ** 2

def squared_error_gradient( x_i, y_i, theta): 
	alpha, beta = theta
	return [-2 * error( alpha, beta, x_i, y_i),  -2 * error( alpha, beta, x_i, y_i) * x_i]

random.seed( 0) 
theta = [random.random(), random.random()] 
alpha, beta = minimize_stochastic( squared_error, squared_error_gradient, predictor, target, theta, 0.0001) 
print (alpha, beta)
print(r_squared( alpha, beta, predictor, target))

plt.scatter(predictor, target,  color='black')
plt.plot(predictor, predictor*beta+alpha, color='blue',
         linewidth=3)
plt.title('Linear regression')
plt.show()
