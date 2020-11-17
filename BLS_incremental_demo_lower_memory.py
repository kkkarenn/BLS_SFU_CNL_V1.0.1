"""
    @authors Zhida Li, Ana Laura Gonzalez Rios, and Guangyu Xu
    @email {zhidal, anag, gxa5}@sfu.ca
    @date Sept. 14, 2019
    @version: 1.0.0
    @description:
                This file creates incremental BLS models based on a 'training_dataset' and 'test_dataset.'
                Each BLS model returns the performance results: accuracy, F-Score, and training time.

    @copyright Copyright (c) Sept. 14, 2019
        All Rights Reserved

    Python code (version 3.6)
"""

######################################################
####### Main file
# Modules of the incremental BLS, RBF-BLS, CFBLS, CEBLS,
# and CFEBLS algorithms
######################################################

# Import the Python libraries
import time
import random
import numpy as np
import sys
from scipy.stats import zscore

from bls.processing.replaceNan import replaceNan
from bls.processing.one_hot_m import one_hot_m
from bls.model.bls_train_fscore_incremental import bls_train_fscore_incremental
from bls.model.rbfbls_train_fscore_incremental import rbfbls_train_fscore_incremental
from bls.model.cfbls_train_fscore_incremental import cfbls_train_fscore_incremental
from bls.model.cebls_train_fscore_incremental import cebls_train_fscore_incremental
from bls.model.cfebls_train_fscore_incremental import cfebls_train_fscore_incremental

from math import ceil

seed = 1; # set the seed for generating random numbers
num_class = 2; # number of the classes

# Load the datasets
# train_dataset = np.loadtxt("./path_to_file/data_train_in_csv_format.csv", delimiter=",");
# test_dataset = np.loadtxt("./pat_to_file/data_test_in_csv_format.csv", delimiter=",");
train_dataset = np.loadtxt("./CICIDS2017_BruteForce_Dataset/ids2017_0407tue_train.csv", delimiter=",");
test_dataset = np.loadtxt("./CICIDS2017_BruteForce_Dataset/ids2017_0407tue_test.csv", delimiter=",");
#train_dataset = np.loadtxt("./CICIDS2017_DoS_Dataset/ids2017_0507wed_train_60.csv", delimiter=",");
#test_dataset = np.loadtxt("./CICIDS2017_DoS_Dataset/ids2017_0507wed_test_40.csv", delimiter=",");

# Normalize training data
train_x = train_dataset[:, 0:train_dataset.shape[1] - 1 ];
train_x = zscore(train_x, axis = 0, ddof = 1); # For each feature, mean = 0 and std = 1 
replaceNan(train_x);                           # Replace "nan" with 0
train_y = train_dataset[:,train_dataset.shape[1]  - 1 : train_dataset.shape[1]];

# Change training labels
inds1 = np.where(train_y == 0);
train_y[inds1] = 2;

# Normalize test data
test_x = test_dataset[:, 0:test_dataset.shape[1] - 1 ];
test_x = zscore(test_x, axis = 0, ddof = 1); # For each feature, mean = 0 and std = 1 
replaceNan(test_x);							 # Replace "nan" with 0
test_y = test_dataset[:, test_dataset.shape[1]  - 1 : test_dataset.shape[1] ];

# Change test labels
inds1 = np.where(test_y == 0);
test_y[inds1] = 2;

train_y = one_hot_m(train_y, num_class);
test_y = one_hot_m(test_y, num_class);

# BLS parameters 
C = 2**-25; # parameter for sparse regularization
s= 0.8;     # the shrinkage parameter for enhancement nodes 

# N1* - the number of mapped feature nodes
# N2* - the groups of mapped features
# N3* - the number of enhancement nodes

N1_bls = 10
N2_bls = 20
N3_bls = 200

N1_rbfbls = 40
N2_rbfbls = 10
N3_rbfbls = 30

N1_cfbls = 10
N2_cfbls = 10
N3_cfbls = 10

N1_cefbls = 10
N2_cefbls = 20
N3_cefbls = 200

N1_cfebls = 15
N2_cfebls = 10
N3_cfebls = 20

epochs = 1; # number of epochs 
inputData = ceil(train_x.shape[0]*0.7)

# BLS parameters for incremental learning
l = 5;     # steps
m2 = 10;   # 20,40, additional enhancement nodes for each step

train_xf = train_x; # the entire training dataset
train_yf = train_y; # the entire training labels

train_x = train_xf[ 0:(int)(inputData) , : ]; # training data at the beginning of the incremental learning
train_y = train_yf[ 0:(int)(inputData) , : ]; # training labels at the beginning of the incremental learning

m = int(ceil((  train_xf.shape[0] - inputData) / l )); # the number of added data points/step

print ("Incremental step is: ", l);

train_err = np.zeros( (1, epochs) );
test_err = np.zeros( (1,epochs) );
train_time = np.zeros( (1,epochs) );
test_time = np.zeros( (1,epochs) );

# # BLS ----------------------------------------------------------------
print("================== BLS (incremental)===========================\n\n");

np.random.seed(seed); # set the seed for generating random numbers
for j in range(0, epochs):

	TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score = \
	bls_train_fscore_incremental(train_x, train_y, train_xf, train_yf, test_x, test_y, 
								s, C, N1_bls, N2_bls, N3_bls, inputData, m, m2, l);

	train_err[0, j] = TrainingAccuracy * 100;
	test_err[0, j] = TestingAccuracy * 100;
	train_time[0, j] = Training_time;
	test_time[0, j] = Testing_time;

bls_test_acc = TestingAccuracy;
bls_test_f_score = f_score;
bls_train_time = Training_time;
bls_test_time = Testing_time;

# # RBF BLS ----------------------------------------------------------------
print ("================== RBF BLS (incremental)===========================\n\n");

np.random.seed(seed);
for j in range(0, epochs):

	TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score = \
	rbfbls_train_fscore_incremental(train_x, train_y, train_xf, train_yf, test_x, test_y, 
								s, C, N1_rbfbls, N2_rbfbls, N3_rbfbls, inputData, m, m2, l);

	train_err[0, j] = TrainingAccuracy * 100;
	test_err[0, j] = TestingAccuracy * 100;
	train_time[0, j] = Training_time;
	test_time[0, j] = Testing_time;


rbfbls_test_acc = TestingAccuracy;
rbfbls_test_f_score = f_score;
rbfbls_train_time = Training_time;
rbfbls_test_time = Testing_time;

# # CFBLS BLS ----------------------------------------------------------------
print ("================== CFBLS BLS (incremental)===========================\n\n");

np.random.seed(seed);
for j in range(0, epochs):

	TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score = \
	cfbls_train_fscore_incremental(train_x, train_y, train_xf, train_yf, test_x, test_y, 
								s, C, N1_cfbls, N2_cfbls, N3_cfbls, inputData, m, m2, l);
    
	train_err[0, j] = TrainingAccuracy * 100;
	test_err[0, j] = TestingAccuracy * 100;
	train_time[0, j] = Training_time;
	test_time[0, j] = Testing_time;
    
cfbls_test_acc = TestingAccuracy;
cfbls_test_f_score = f_score;
cfbls_train_time = Training_time;
cfbls_test_time = Testing_time;

# # CEBLS BLS ----------------------------------------------------------------
print ("================== CEBLS BLS (incremental)===========================\n\n");

np.random.seed(seed);
for j in range(0, epochs):
    
	TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score = \
	cebls_train_fscore_incremental(train_x, train_y, train_xf, train_yf, test_x, test_y, 
								s, C, N1_cefbls, N2_cefbls, N3_cefbls, inputData, m, m2, l);
    
    
	train_err[0, j] = TrainingAccuracy * 100;
	test_err[0, j] = TestingAccuracy * 100;
	train_time[0, j] = Training_time;
	test_time[0, j] = Testing_time;
    
cebls_test_acc = TestingAccuracy;
cebls_test_f_score = f_score;
cebls_train_time = Training_time;
cebls_test_time = Testing_time;

# # CFEBLS BLS ----------------------------------------------------------------
print ("================== CFEBLS BLS (incremental)===========================\n\n");
np.random.seed(seed);
    
for j in range(0, epochs):
    
	TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score = \
	cfebls_train_fscore_incremental(train_x, train_y, train_xf, train_yf, test_x, test_y, 
								s, C, N1_cfebls, N2_cfebls, N3_cfebls, inputData, m, m2, l);
    
    
	train_err[0, j] = TrainingAccuracy * 100;
	test_err[0, j] = TestingAccuracy * 100;
	train_time[0, j] = Training_time;
	test_time[0, j] = Testing_time;
    
cfebls_test_acc = TestingAccuracy;
cfebls_test_f_score = f_score;
cfebls_train_time = Training_time;
cfebls_test_time = Testing_time;


print ("BLS Test Acc: ", bls_test_acc*100, " fscore: ", bls_test_f_score*100, "Training time: ", bls_train_time);
print ("RBF-BLS Test Acc: ", rbfbls_test_acc*100, " fscore: ", rbfbls_test_f_score*100, "Training time: ", rbfbls_train_time);
print ("CFBLS Test Acc: ", cfbls_test_acc*100, " fscore: ", cfbls_test_f_score*100, "Training time: ", cfbls_train_time);
print ("CEBLS Test Acc: ", cebls_test_acc*100, " fscore: ", cebls_test_f_score*100, "Training time: ", cebls_train_time);
print ("CFEBLS Test Acc: ", cfebls_test_acc*100, " fscore: ", cfebls_test_f_score*100, "Training time: ", cfebls_train_time);
print("End of the execution");
