# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:43:00 2019

@author: wangqi44
"""

from datetime import datetime
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import pickle as pkl
import tensorflow as tf

f = open('../../data/processed_data/SGP.pickle', "rb")
data = pkl.load(f)
f.close()


# Training Data
data_rp = data["X_train_rp"]
data_sp = data["X_train_sp"]
choice_rp = data["Y_train_rp"]
choice_sp = data["Y_train_sp"]

# Testing Data
data_test_rp = data["X_test_rp"]
data_test_sp = data["X_test_sp"]
choice_test_rp = data["Y_test_rp"]
choice_test_sp = data["Y_test_sp"]

rpsp = 'rp'
data = data_sp
choice = choice_sp
data_test = data_test_sp
choice_test = choice_test_sp
data = data_rp
choice = choice_rp
data_test = data_test_rp
choice_test = choice_test_rp

x_train_nn = np.array(data)
choice_train_nn = np.array(choice)
x_test_nn = np.array(data_test)
choice_test_nn = np.array(choice_test)

feature_columns = [tf.feature_column.numeric_column("x", shape = (np.size(data, axis=1)))]
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x":x_train_nn},
    y = choice_train_nn,
    num_epochs=500,
    batch_size = 300,
    shuffle = True
)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x": x_test_nn},
    y = choice_test_nn,
    num_epochs=1,
    shuffle=False
)

train_input_fn_eval = tf.estimator.inputs.numpy_input_fn(
    x = {"x":x_train_nn},
    y = choice_train_nn,
    num_epochs=1, 
    shuffle=False
)

ll1 = [1e-20, 1e-4, 1e-2, 5e-1]
ll2 = [1e-20, 1e-4, 1e-2, 5e-1]
ll1 = [1e-2, 5e-1, 1e-20, 1e-4]
ll2 = [1e-2, 5e-1, 1e-20, 1e-4]

if not os.path.exists("temp_result_sep_nn_"+rpsp+".txt"):
    outfile = open("temp_result_sep_nn_"+rpsp+".txt", 'a')
    outfile.write("hiddenunits, layers, l1,l2, tr_accuracy,test_accuracy\n")
    outfile.close()
count = 0
hus = [[50,	100,	200,	25	,25],
[25,	200,	100	,25	,100],
[25	,100	,25,	25	,50],
[200,	25,	25,	200	,50],
[200,	100,	100	,100	,50],
[100,	200	,50	,25	,50],
[100,	50	,25	,25	,100],
[50,	50	,100,	100	,200],
[50,	200	,100,	200	,25],
[100,	200	,50	,50	,50],
[200,	25	,25	,100	,25],
[200,	100	,200,	50	,50],
[100,	25	,200,	200	,50],
[200,	200	,25	,50	,25],
[25,	25	,50,	50	,50],
[25	,50,	100	,100	,50]]

for l1 in ll1:
    for l2 in ll2:
        hu = hus[count]
        count += 1
        '''hu = []
        for i in range(5):
            hu.append(np.random.choice([25, 50, 100, 200]))'''
        classifier = tf.estimator.DNNClassifier(
                feature_columns = feature_columns,
                hidden_units = hu,
                optimizer = tf.train.ProximalAdagradOptimizer(
                        learning_rate = 0.001, 
                        l1_regularization_strength=l1, 
                        l2_regularization_strength=l2), 
                n_classes = 5,
                dropout = 0.5
                )
    
        print('Start Training', datetime.now())
        classifier.train(input_fn=train_input_fn)
        print('Training Concluded', datetime.now())
        
            
        # Training Accuracy
        
        results_train = classifier.evaluate(input_fn=train_input_fn_eval)
        
        print('Training Accuracy:' , results_train["accuracy"])
        
        # Testing Accuracy
        results_test = classifier.evaluate(input_fn=test_input_fn)
        
        print('Testing Accuracy:', results_test["accuracy"])
        
        outfile = open("temp_result_sep_nn_"+rpsp+".txt", 'a')
        outfile.write("%s, %.2E, %.2E, %.4f, %.4f \n" % \
                      (str(hu), l1, l2, results_train['accuracy'], results_test['accuracy']))
        outfile.close()
    
