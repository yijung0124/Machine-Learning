import numpy as np
import pandas as pd
import csv
import os
import copy
import matplotlib.pyplot as plt


##part1
pwd = os.getcwd()
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
# learning_list = [pow(10, 0),pow(10, -1),pow(10, -2),pow(10, -3),pow(10, -4),pow(10, -5),pow(10, -6),pow(10, -7)]
<<<<<<< HEAD
lamda = 0.001
learning = pow(10, -1)
=======

learning = pow(10, -5)
>>>>>>> 208aa74d577d3b719bd9bb6c9123897cdfa7fa31
normalg_list = list()



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
    sse = 0
    sum_up = 0
    N = x.shape[0]      #we need to know how many data in each column(How many rows)

    for i in range(0, N):
        sse += (y[i] - np.dot(w, x[i]))**(2)
        sum_up += 2 * (np.dot(w, x[i]) - y[i]) * x[i] + 2 * lamda * w
    print("sse:",sse)
    normalg_list.append(sse)
    return sum_up


def grad_descent (x, y, learning):
    """
    The grad_descent function of different learning rate and fixed lamda
    w: weight
    learning: learning rate
    converage: converage limit value
    """ 

    w = np.zeros(22)
    converage=40
    i=0
    for runs in range(1000000):
        gradient = grad(w, x, y, lamda)
        w = w - (learning * gradient)
        normalg= np.linalg.norm(gradient)
        print("normalg: ", normalg)
        #print("gradient: ", gradient)
        if np.isinf(normalg):
            print(normalg_list)
            break
        #normalg_list.append(gradient)
        if normalg <= converage:
            print("normalg <= converage!!!")
            break
  
    print("total run: ",i)
    print("w: ",w )
    return w


def test_y_value(w, x):
    '''
        This function is for finding y value for test. file
        w: Best w value
        x: test. file without price column
        y: use y value from train data or validation data
    '''

    pred_y = np.array([])           #store pred_value

    for i in x:
        value = np.dot(w, i)    
        pred_y = np.append(pred_y, value)

    return pred_y


def cross_comparison_dev(w, true_dev_y):
    pred_dev_y = test_y_value(w, normalized_dev_data)
    sum_difference_y = float()
    for (ea_true_dev_y, ea_pred_dev_y )in zip(true_dev_y, pred_dev_y):
        difference_y = abs(ea_true_dev_y - ea_pred_dev_y)
        sum_difference_y += difference_y

def predict_test_y():
    w = [-1.86290481,0.1075248,-0.38702919,-0.13091035,-0.31506836,4.05686198,4.88834682,0.15583834,0.69730991,3.60116205,2.40249443,0.70713441,7.44499907,5.52841333,2.99076192,-2.74338666,0.65187757,-1.08878388,3.77410322,-1.970244,4.60586131,-0.19908633]
    pred_test_y = test_y_value(w, normalized_test_data)
    pty = pred_test_y.tolist()
    print(pty)
    with open('new_predict_test_y.csv', 'a', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        writer.writerow(pty)


def count_percentage(data, title):
    print("Count percentage of examples for ",title," : ")
    originalList = data.tolist()
    total_len = len(originalList)
    diff_num = set(originalList) 
    fracs = list()
    # tmp_table = list()
    with open('count_percentage.csv', 'a', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        writer.writerow([title])


    for ea_diff_num in diff_num:

        fracs.append(ea_diff_num)
        fracs.append(100*originalList.count(ea_diff_num)/total_len)
        print(ea_diff_num, ":  ",100*originalList.count(ea_diff_num)/total_len)

        with open('count_percentage.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fracs)

        del fracs[:]
        


    
if __name__ == "__main__":
    y_train_data, y_dev_data = process_columns()
    grad_descent(normalized_train_data, y_train_data, learning)

    # predict price 
    predict_test_y()

    #cross comparison to get difference_y
    #cross_comparison_dev(input_w, y_dev_data)


    # number = [0, 1, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    # b = train[:1].T
    # a = train[1:].T
    # table = list()
    # for i in number: 
    #     feature_name = b[i]   
    #     float_value = a[i].astype(float)
    #     max_value = np.max(float_value)
    #     min_value = np.min(float_value)
    #     mean_value = (sum(float_value))/10000
    #     std_value = np.std(float_value)
    #     # print('feature:', feature_name, 'max_value:', max_value, 'min_value:',  min_value, 'mean_value:', mean_value, 'std_value:', std_value )
    #     tmp = [feature_name, max_value, min_value, mean_value, std_value]
    #     table.append(tmp)
    # with open('range_table.csv', 'a', newline='') as csvfile:
        
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["feature", "max_value", "min_value", "mean_value", "std_value"])
    #     writer.writerows(table)

    # plt.plot(normalg_list)
    # plt.savefig(pwd+"/pic.png")
    # plt.show()
    # del normalg_list[:]
    # w = [0.7929263,0.24809381,0.19291893,0.13768568,0.86682762,1.22872843,1.1341017,0.08855074,0.90892737,0.580687,1.44196388,0.64715359,1.9693417,1.2604114,0.73537896,0.2428598,0.685939,-0.04664689,1.9124737,0.12728947,1.6342164,0.11902113]
