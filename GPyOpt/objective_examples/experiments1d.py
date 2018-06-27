# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import matplotlib.pyplot as plt
import numpy as np


class function1d:
	'''
	This is a benchmark of unidimensional functions interesting to optimize. 
	:param bounds: the box constraints to define the domain in which the function is optimized.
	'''
	def plot(self,bounds=None):
		if bounds is  None: bounds = self.bounds
		X = np.arange(bounds[0][0], bounds[0][1], 0.01)
		Y = self.f(X)
		plt.plot(X, Y, lw=2)
		plt.xlabel('x')
		plt.ylabel('f(x)')
		plt.show()

class forrester(function1d):
	'''
	Forrester function. 
	
	:param sd: standard deviation, to generate noisy evaluations of the function.
	'''
	def __init__(self,sd=None):
		self.input_dim = 1		
		if sd==None: self.sd = 0
		else: self.sd=sd
		self.min = 0.78 		## approx
		self.fmin = -6 			## approx
		self.bounds = [(0,1)]

	def f(self,X):
		X = X.reshape((len(X),1))
		n = X.shape[0]
		fval = ((6*X -2)**2)*np.sin(12*X-4)
		if self.sd ==0:
			noise = np.zeros(n).reshape(n,1)
		else:
			noise = np.random.normal(0,self.sd,n).reshape(n,1)
		return fval.reshape(n,1) + noise



class  synthetic(function1d):


	def __init__(self):
		self.min = (45, 45.5)
		self.fmin = -200
		self.bounds = [(0, 100)]
	def f(self,X):
		X = X.reshape((len(X),1))
		n = X.shape[0]
		Y = np.zeros((n , 1))
		for index in range(n):
			x= X[index]
			if x > 35.0 and x < 35.5:
				Y[index] = -100
			elif x>45.0 and x < 45.5:
				Y[index] = -200
			else:
				Y[index] = 200 * np.sin(8 * np.pi * x / 50) * np.sin(3 * x / 100)
		return Y

