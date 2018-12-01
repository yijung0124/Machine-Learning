import numpy as np
from math import log
import matplotlib.pyplot as plt
import csv


##########Read File###########
def y_data(filename):
	'''
		This function is for adding y value in list
	'''
	file = open(filename , 'r')
	add_y = []
	y = []
	a = csv.reader(file)
	for row in a : 
		add_y.append(row[0])
	for iy in add_y:
		integer = int(iy)
		if integer == 3:
			y.append(float(1))
		else:
			y.append(float(-1))
	y_to_array = np.array(y)
	return y_to_array

def x_data(filename):
	'''
		This function is for x value without y value
	'''
	file = open(filename, 'r') 
	add_x = []
	for row in file:
		tmp = []						#each row creates a list
		for x in row.split(','):		
			tmp.append(float(x.strip()))		
		add_x.append(tmp[1:])
	x_to_array = np.array(add_x)			# x list convert to x array

	return x_to_array


#############Perceptron##############
def Perceptron(x, y):
	'''
		
	'''
	w = np.zeros(len(x[0]))		#number of values in Train.csv
	it = 0					#iterate
	u = 0 					#initial x dot w.T
	N = x.shape[0]		
	accuracy = 0

	while it < 16:
		count = 0
		for i in range(0, N):
			u = np.dot(x[i], w.T)
			if (y[i]*u) <= 0:
				count = count + 1
				w = w + y[i]*x[i]
		accuracy = (N-count)/N
		print(accuracy)			
		it = it + 1
	return w


############Main Function############

y_array = y_data('pa2_train.csv')
x_array = x_data('pa2_train.csv')
Perceptron(x_array, y_array)
