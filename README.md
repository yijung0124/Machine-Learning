# CS534
****
##<IA2>
1.  In IA2-part1.py, set the iters = 15, run Perceptron_Validation to print out the accuracies for the train and validation at the end of each iteration. Use the validation accuracy to decide the test number for iters, and run test_value to generate the prediction file oplabel.csv.
2.  In IA2-part2.py, set the iters = 15,run Avg_Perceptron to print out the train and validation accuracies.
3.  In IA2-part3.py, run both Kernel_Perceptron_Train and Kernel_Perceptron_Valid to print out accuracies, but run Kernel_Perceptron_Train first to get the current alpha to predict validation set. Use the best alpha to predict the test data-set, and output predicted file as kplabel.csv.

****
##<IA1>
1. Source the server python3.5 env, we need to install pandas lib.
   pip install pandas
   pip install --upgrade pandas
2. For running part1 program
   python3.5 IA1.py
     in this program file includes two function:
     --grad_descent(normalized_train_data, y_train_data, learning)
     this function generates the weight matrix
     --predict_test_y()
     this function to predict the test y value by using the weight we found
3. For running part2 program
   python3.5 IA1-part2.py
4. For running part3 program
   python3.5 IA1-part3.py
