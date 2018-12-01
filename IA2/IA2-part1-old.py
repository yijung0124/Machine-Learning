import numpy as np
from math import log
import matplotlib.pyplot as plt
import csv
import math


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
		tmp.append(float(1))		
		add_x.append(tmp[1:])
	x_to_array = np.array(add_x)			# x list convert to x array

	return x_to_array

def test_x_data(filename):
	'''
		This function is for x value 
	'''
	file = open(filename, 'r') 
	add_x = []
	for row in file:
		tmp = []						#each row creates a list
		for x in row.split(','):		
			tmp.append(float(x.strip()))
		tmp.append(float(1))		
		add_x.append(tmp[:])
	x_to_array = np.array(add_x)			# x list convert to x array
	# print("x_to_array",x_to_array.shape)
	return x_to_array


#############Perceptron##############
def Perceptron(x, y, iters):
	'''
		
	'''
	w = np.zeros(len(x[0]))		#number of values in Train.csv
	it = 0					#iterate
	u = 0 					#initial x dot w.T
	N = x.shape[0]		
	accuracy = 0

	while it < iters:
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

###########Test value################
def test_value(w, x):

	y = np.zeros(x.shape[0])	
	for i in range(0, x.shape[0]):
		y[i] = np.sign(np.dot(x[i], w.T))
	return y


############Main Function############


# y_array = y_data('pa2_train.csv')
# x_array = x_data('pa2_train.csv')
# Perceptron(x_array, y_array)


#train.csv
t_y_array = y_data('pa2_train.csv')
t_x_array = x_data('pa2_train.csv')
#valid.csv
y_array = y_data('pa2_valid.csv')
x_array = x_data('pa2_valid.csv')
#test.csv
test_x_array = test_x_data('pa2_test_no_label.csv')

# print("=========train==========")
# Perceptron(t_x_array, t_y_array, 16)
# print("=========valid==========")
# Perceptron(x_array, y_array, 16)

# print("====Find best iters=====valid==========")
# Perceptron(x_array, y_array, 31)


# use the val's y_array and x_array
w = Perceptron(x_array, y_array, 30)
pred_y = test_value(w, test_x_array)

rows = pred_y.shape[0]
with open('oplabel.csv', 'a', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        for i in range(0, rows):
            writer.writerow([pred_y[i]])

