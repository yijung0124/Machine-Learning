import numpy as np
import pandas as pd
import csv
import os
import copy
import matplotlib.pyplot as plt

##part1

train = pd.read_csv('PA1_train.csv', sep=',',header=None)
train = train.values
test = pd.read_csv('PA1_test.csv', sep=',',header=None)
test = test.values
dev = pd.read_csv('PA1_dev.csv', sep=',',header=None)
dev = dev.values
raw_train_data = np.zeros((10000, 22))  ## take out id and price 
raw_test_data = np.zeros((6000, 22))  ## take out id 
raw_dev_data = np.zeros((5597, 22))  ## take out id and price 
normalized_train_data = np.zeros((10000, 22))  ## take out id and price 
normalized_test_data = np.zeros((6000, 22))  ## take out id 
normalized_dev_data = np.zeros((5597, 22))  ## take out id and price 
y_train_data = np.zeros((10000, ))
y_dev_data = np.zeros((5597, ))

lamda = 10**(-2)
sse_list = list()
iteration_list = list()



def split_date(cut_head_data, whichForm):
    split_date_data = copy.deepcopy(cut_head_data)
    if whichForm == 'train':
        sd_data = np.zeros((10000,3))
    if whichForm == 'test':
        sd_data = np.zeros((6000,3))
    if whichForm == 'dev':
        sd_data = np.zeros((5597,3))
        
    for idx_r, ea_date_str in enumerate(split_date_data):
        data_features = ea_date_str.split("/")
        for idx in range(0,3):
            sd_data[idx_r,idx] = data_features[idx-1]
        idx_r += 1
    h_data_set = np.hsplit(sd_data,3)
    return h_data_set

def add_in_arrays(count_col, data, min_array, max_array):
    """
    Add the max and the min in an array.
    """
    max_array[count_col] = np.max(data)
    min_array[count_col] = np.min(data)

def norm_data(ea_col, count_col, cut_head_data, min_array, max_array, whichForm):
    """
    Normalize data.
    """
    new_data = (cut_head_data - min_array[count_col]) / (max_array[count_col] - min_array[count_col])
    
    if whichForm == 'train':
        if ea_col == 2:
            new_data =  new_data.reshape((10000,))
        normalized_train_data[:,count_col] = new_data

    if whichForm == 'test':
        if ea_col == 2:
            new_data =  new_data.reshape((6000,))
        normalized_test_data[:,count_col] = new_data
        
    if whichForm == 'dev':
        if ea_col == 2:
            new_data =  new_data.reshape((5597,))
        normalized_dev_data[:,count_col] = new_data


def process_columns():
    """
    Process both test.csv and train.csv 's columns and normalize them
    The final normalized data will store in normalized_train_data and normalized_test_data (without 'id' and 'price' columns )
    """
    
    count_col = 0
    
    # Run through every col in train.csv
    whichForm = 'train'
    min_array = np.zeros((train.shape[1],))
    max_array = np.zeros((train.shape[1],))

    for ea_col in range(train.shape[1]):
        
        orig_data = train[:,ea_col]
        
        cut_head_data = copy.deepcopy(orig_data)
        cut_head_data = cut_head_data[1:]
        
        if ea_col == 2:
            date_data = split_date(cut_head_data, whichForm)
            for ea_date_data in date_data:
                raw_train_data[:,count_col] = ea_date_data.reshape((10000,))
                add_in_arrays(count_col, ea_date_data, min_array, max_array)
                norm_data(ea_col, count_col, ea_date_data, min_array, max_array, whichForm)
                count_col += 1
        elif ea_col == 0:
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            normalized_train_data[:, 0] = cut_head_data
            raw_train_data[:,0] = cut_head_data.reshape((10000,))
            count_col += 1
        elif ea_col == 1:
            pass
        elif ea_col == 21:
            cut_head_data = cut_head_data.astype(float)
            y_train_data = cut_head_data
        else:
            cut_head_data = cut_head_data.astype(float)
            raw_train_data[:,count_col] = cut_head_data.reshape((10000,))
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            norm_data(ea_col, count_col, cut_head_data, min_array, max_array, whichForm)
            count_col += 1

    ##########################################################################
    
    count_col = 0
    
    # Run through every col in test.csv
    whichForm = 'test'
    min_array = np.zeros((test.shape[1]+1,))
    max_array = np.zeros((test.shape[1]+1,))

    for ea_col in range(test.shape[1]):
        orig_data = test[:,ea_col]
        
        cut_head_data = copy.deepcopy(orig_data)
        cut_head_data = cut_head_data[1:]
        
        if ea_col == 2:
            date_data = split_date(cut_head_data, whichForm)
            for ea_date_data in date_data:
                raw_test_data[:,count_col] = ea_date_data.reshape((6000,))
                add_in_arrays(count_col, ea_date_data, min_array, max_array)
                norm_data(ea_col, count_col, ea_date_data, min_array, max_array, whichForm)
                count_col += 1
        elif ea_col == 0:
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            normalized_test_data[:, 0] = cut_head_data
            raw_test_data[:,0] = cut_head_data.reshape((6000,))
            count_col += 1
        elif ea_col == 1:
            pass
        else:
            cut_head_data = cut_head_data.astype(float)
            raw_test_data[:,count_col] = cut_head_data.reshape((6000,))
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            norm_data(ea_col, count_col, cut_head_data, min_array, max_array, whichForm)
            count_col += 1


    ##########################################################################
    
    count_col = 0
    
    # Run through every col in dev.csv
    whichForm = 'dev'
    min_array = np.zeros((dev.shape[1],))
    max_array = np.zeros((dev.shape[1],))

    for ea_col in range(dev.shape[1]):
        
        orig_data = dev[:,ea_col]
        
        cut_head_data = copy.deepcopy(orig_data)
        cut_head_data = cut_head_data[1:]
        
        if ea_col == 2:
            date_data = split_date(cut_head_data, whichForm)
            for ea_date_data in date_data:
                raw_dev_data[:,count_col] = ea_date_data.reshape((5597,))
                add_in_arrays(count_col, ea_date_data, min_array, max_array)
                norm_data(ea_col, count_col, ea_date_data, min_array, max_array, whichForm)
                count_col += 1
        elif ea_col == 0:
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            normalized_dev_data[:, 0] = cut_head_data
            raw_dev_data[:,0] = cut_head_data.reshape((5597,))
            count_col += 1
        elif ea_col == 1:
            pass
        elif ea_col == 21:
            cut_head_data = cut_head_data.astype(float)
            y_dev_data = cut_head_data
        else:
            cut_head_data = cut_head_data.astype(float)
            raw_dev_data[:,0] = cut_head_data.reshape((5597,))
            add_in_arrays(count_col, cut_head_data, min_array, max_array)
            norm_data(ea_col, count_col, cut_head_data, min_array, max_array, whichForm)
            count_col += 1
            
    return y_train_data, y_dev_data




def grad(w, x, y, lamda):   
    """
    The gradient of the linear regression with l2 regularization cost function
    x:input dataset
    y:output dataset
    lamda:regularization factor
    """

    sum_up = 0
    sum_sse = 0
    N = x.shape[0]      #we need to know how many data in each column(How many rows)

    for i in range(0, N):
        sum_up = 2 * (np.dot(w, x[i]) - y[i]) * x[i] + 2 * lamda * w
        sum_sse = (((np.dot(w, x[i]) - y[i]))**(2)) + sum_sse
        sse_list.append(sum_sse)
    print(sum_sse)
       
    return sum_up


def diff_lamda(x, y, lamda):
    '''
    The regularization of different lamda values and fixed learning rate
    x:input dataset
    y:output dataset
    lamda:regularization factor
    rate:learning rat
    '''
    
    w = np.zeros(22)   #initial w
    rate = 10**(-5) #fixed rate
    converage=300

    for runs in range(1000000):
        E = grad(w, x, y, lamda)
        w = w - ( rate * E)
        normalg= np.linalg.norm(E)
        if np.isinf(normalg):
            print("normalg goes to inf, goning to break the loop.")
            break
        if normalg <= converage:
            print("normalg <= converage!!!")
            break
        iteration_list.append(runs)
    print("w: ", w)        
    print("i: ", runs)
    return w




    
if __name__ == "__main__":
    y_train_data, y_dev_data = process_columns()
    diff_lamda(normalized_train_data, y_train_data, lamda)
    plt.plot(sse_list, iteration_list)
    plt.ylabel("Number of iteration")
    plt.xlabel("SSE")
    plt.savefig(pwd+"train_part2-10^-9.png")

    