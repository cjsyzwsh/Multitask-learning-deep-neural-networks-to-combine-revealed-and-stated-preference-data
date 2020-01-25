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
data_rp["rp"] = 1
data_sp["rp"] = 0
choice_rp = data["Y_train_rp"]
choice_sp = data["Y_train_sp"]
choice = np.concatenate((choice_rp, choice_sp))

# Testing Data
data_test_rp = data["X_test_rp"]
data_test_sp = data["X_test_sp"]
data_test_rp["rp"] = 1
data_test_sp["rp"] = 0
choice_test_rp = data["Y_test_rp"]
choice_test_sp = data["Y_test_sp"]
choice_test = np.concatenate((choice_test_rp, choice_test_sp))

# Pool Data Together
data = pd.concat([data_rp, data_sp])
data_test = pd.concat([data_test_rp, data_test_sp])

x_train_nn = np.array(data)
choice_train_nn = choice
x_test_nn = np.array(data_test)
choice_test_nn = choice_test

feature_columns = [tf.feature_column.numeric_column("x", shape = (np.size(data, axis=1)))]
ll1 = [1e-20, 1e-4, 1e-2, 5e-1]
ll2 = [1e-20, 1e-4, 1e-2, 5e-1]

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

if not os.path.exists("temp_result_pooled_nn.txt"):
    outfile = open('temp_result_pooled_nn.txt', 'a')
    outfile.write("hu, l1,l2, tr_overall_accuracy,tr_rp_accuracy, tr_sp_accuracy,overall_accuracy, rp_accuracy,sp_accuracy\n")
    outfile.close()

for l1 in [1e-20]:
    for l2 in [5e-1]:
        hu = [25, 50, 25, 50, 50]
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
        results_train = classifier.evaluate(input_fn=train_input_fn)
        
        print('Start Training', datetime.now())
        classifier.train(input_fn=train_input_fn)
        print('Training Concluded', datetime.now())
    
        # Training Accuracy
        
        results_train = classifier.evaluate(input_fn=train_input_fn)
            
        data['predicted_choice'] = [i['class_ids'] for i in classifier.predict(input_fn=train_input_fn_eval)]
        data['choice'] = choice_train_nn
        
        rp_acc_train = np.sum(data[data['rp'] == 1]['predicted_choice'] == data[data['rp'] == 1]['choice']) / len(data[data['rp'] == 1])
        sp_acc_train = np.sum(data[data['rp'] == 0]['predicted_choice'] == data[data['rp'] == 0]['choice']) / len(data[data['rp'] == 0])
            
        # Testing Accuracy
        results_test = classifier.evaluate(input_fn=test_input_fn)
          
        data_test['predicted_choice'] = [i['class_ids'] for i in classifier.predict(input_fn=test_input_fn)]
        data_test['choice'] = choice_test_nn
        
        rp_acc_test = np.sum(data_test[data_test['rp'] == 1]['predicted_choice'] == data_test[data_test['rp'] == 1]['choice']) / len(data_test[data_test['rp'] == 1])
        sp_acc_test = np.sum(data_test[data_test['rp'] == 0]['predicted_choice'] == data_test[data_test['rp'] == 0]['choice']) / len(data_test[data_test['rp'] == 0])
        
        outfile = open('temp_result_pooled_nn.txt', 'a')
        outfile.write("%s, %.2E, %.2E, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n" % \
                      (str(hu), l1, l2, results_train['accuracy'], rp_acc_train, sp_acc_train, results_test['accuracy'], rp_acc_test, sp_acc_test))
        outfile.close()
    
