"""
Created on Mon Jun 24 13:16:50 2019

util_new

@author: shenhao
"""

import os
import numpy as np
import pandas as pd
import pickle
#import statsmodels.api as sm
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import copy
from sklearn.preprocessing import StandardScaler

#cd "/Users/shenhao/Dropbox (MIT)/Shenhao_Jinhua (1)/7_ml_structure/code"

def load_data():
    with open('../data/processed_data/SGP.pickle', 'rb') as f:
        SGP = pickle.load(f)
    # rp
    Y_rp_train = SGP['Y_train_rp'].values
    X_rp_train = SGP['X_train_rp'].values
    Y_rp_test = SGP['Y_test_rp'].values
    X_rp_test = SGP['X_test_rp'].values
    # sp
    Y_sp_train = SGP['Y_train_sp'].values
    X_sp_train = SGP['X_train_sp'].values
    Y_sp_test = SGP['Y_test_sp'].values
    X_sp_test = SGP['X_test_sp'].values
    
    return Y_rp_train, Y_rp_test, Y_sp_train, Y_sp_test, X_rp_train, X_rp_test, X_sp_train, X_sp_test

def compute_elasticity(X_sp, x, var, prob, sess):
    # (deprecated) compute elasticity from perturbing inputs
    with open('../data/processed_data/SGP_raw.pickle', 'rb') as f:
        SGP = pickle.load(f)
    # sp
    X_sp_train = SGP['X_train_sp'].values
    names = ['walk_walktime',
             'bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt',
             'ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt',
             'av_cost', 'av_waittime', 'av_ivt',
             'drive_cost', 'drive_walktime', 'drive_ivt',
             'age', 'inc', 'edu',
             'male','young_age', 'old_age',
             'low_edu', 'high_edu',
             'low_inc', 'high_inc',
             'full_job']
    X_sp_train = pd.DataFrame(X_sp_train, columns = names)

    standard_vars = ['walk_walktime',
                 'bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt',
                 'ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt',
                 'av_cost', 'av_waittime', 'av_ivt',
                 'drive_cost', 'drive_walktime', 'drive_ivt',
                 'age', 'inc', 'edu']
    non_standard_vars = ['young_age', 'old_age',
                     'low_edu', 'high_edu',
                     'low_inc', 'high_inc',
                     'full_job']
    X_sp_train_standard = X_sp_train[standard_vars]
    X_sp_train_nonstandard = X_sp_train[non_standard_vars]

    grad_sp = tf.gradients(prob_sp[:, 1], X_sp)
    x_avg_standard = np.array(X_sp_train_standard.mean()) #np.reshape(np.mean(X_sp_train_standard, axis = 0), (1, np.size(X_sp_train_standard, axis=1)))
    x_avg_nonstandard = np.array(X_sp_train_nonstandard.mean()) #np.reshape(np.mean(X_sp_train_nonstandard, axis = 0), (1, np.size(X_sp_train_nonstandard, axis=1)))

    scaler = StandardScaler().fit(X_sp_train_standard)

    elas = []
    prob = sess.run(prob_sp, feed_dict={X_sp: X_sp_train})
    for idx in var:
        x_feed = x_avg_standard
        #grad.append(sess.run(grad_sp, feed_dict={X_sp: x_feed})[0])
        x_feed[:,idx] = x_avg_standard[:,idx] * 1.005
        x_feed = scaler.transform(x_feed)
        x_feed = np.insert(x_feed, -1, x_avg_nonstandard, axis = 1)
        prob_1 = sess.run(prob, feed_dict={X_sp: x_feed})[0]

        x_feed = x_avg_standard
        x_feed[:,idx] = x_avg_standard[:,idx] * 0.995
        x_feed = scaler.transform(x_feed)
        x_feed = np.insert(x_feed, -1, x_avg_nonstandard, axis = 1)
        prob_2 = sess.run(prob, feed_dict={X_sp: x_feed})[0]

        elas.append(prob_1 - prob_2)

    return elas

def compute_prob_curve(X_sp, x, var, prob_sp, sess):

    x_avg = np.mean(x, axis = 0)
    x_feed = np.repeat(x_avg, len(x)).reshape(len(x), np.size(x, axis=1), order='F')
    #print(x_avg)
    choice_prob = []
    for idx in var:
        #x_feed[:,idx] = x_avg[0,idx]
        x_feed[:,idx] = x[:,idx]
        temp = sess.run(prob_sp, feed_dict={X_sp: x_feed})
        choice_prob.append(temp)
        #print(x_feed[5:10,idx-1:idx+2])
        #print(temp[5:10, :])
        #print(prob.eval(feed_dict={X_sp: x_feed})[:5, :])
        x_feed[:,idx] = x_avg[idx]
        #x_feed[:,idx] = x[:,idx]

    return choice_prob

def standard_shared_layer(input_rp, input_sp, n_hidden, name):
    '''
    Add one more standard shared layer in DNN.
    W,b are two shared parameters.
    '''
#    print(tf.shape(input_rp)[1])
#    print(tf.shape(input_sp)[1])
#    assert tf.shape(input_rp)[1] == tf.shape(input_sp)[1]
    
    W = tf.Variable(tf.random_normal([n_hidden, n_hidden]), dtype = tf.float32, name = name+'_W')
    b = tf.Variable(tf.random_normal([n_hidden]), dtype = tf.float32, name = name+'_bias')
    
    hidden_shared_rp = tf.add(tf.matmul(input_rp, W), b)
    hidden_shared_rp = tf.nn.relu(hidden_shared_rp)
    hidden_shared_sp = tf.add(tf.matmul(input_sp, W), b)
    hidden_shared_sp = tf.nn.relu(hidden_shared_sp)
    return hidden_shared_rp,hidden_shared_sp
    
def standard_specific_layer(input_rp, input_sp, n_hidden, name):
    '''
    Add one more standard domain-specific layer in DNN.
    W,b are two shared parameters.
    '''
    W_rp = tf.Variable(tf.random_normal([n_hidden, n_hidden]), dtype = tf.float32, name = name+'_rp_W')
    b_rp = tf.Variable(tf.random_normal([n_hidden]), dtype = tf.float32, name = name+'_rp_bias')
    W_sp = tf.Variable(tf.random_normal([n_hidden, n_hidden]), dtype = tf.float32, name = name+'_sp_W')
    b_sp = tf.Variable(tf.random_normal([n_hidden]), dtype = tf.float32, name = name+'_sp_bias')
    
    hidden_specific_rp = tf.add(tf.matmul(input_rp, W_rp), b_rp)
    hidden_specific_rp = tf.nn.relu(hidden_specific_rp)    
    hidden_specific_sp = tf.add(tf.matmul(input_sp, W_sp), b_sp)
    hidden_specific_sp = tf.nn.relu(hidden_specific_sp)
    return hidden_specific_rp,hidden_specific_sp

def obtain_mini_batch_rpsp(X_rp, Y_rp, X_sp, Y_sp, n_mini_batch):
    '''
    Return mini_batch
    '''
    N_rp, D_rp = X_rp.shape                     
    N_sp, D_sp = X_sp.shape                     
    
    index_rp = np.random.choice(N_rp, size = n_mini_batch)     
    index_sp = np.random.choice(N_sp, size = n_mini_batch)     

    X_rp_batch = X_rp[index_rp, :]
    Y_rp_batch = Y_rp[index_rp]
    X_sp_batch = X_sp[index_sp, :]
    Y_sp_batch = Y_sp[index_sp]
    return X_rp_batch, Y_rp_batch, X_sp_batch, Y_sp_batch

def build_nn(Y_rp_train, Y_rp_test, Y_sp_train, Y_sp_test, 
             X_rp_train, X_rp_test, X_sp_train, X_sp_test,
             M_share, M_specific, n_hidden,  
             l_shared, l_rp_sp, l_sp,
             sp_rp_N_ratio, 
             n_iterations, n_mini_batch, MODEL_NAME = 'model', temperature = None,
             K_rp = 4, K_sp = 5, D = 25):
    '''
    build the multitask learning DNN and train it.
    Some inputs are not explicitly specified: X_rp_train, Y_rp_train, etc.
    L1 should be true of false; L1_const is the strength of regularization
    '''
    import tensorflow as tf

    assert M_share >= 0, "M_share should be larger than zero."
    assert M_specific >= 1, "M_specific should be larger than one."

    #n_mini_batch = 100

    # 0. reset first
    tf.reset_default_graph() 
    
    # 1. model construction with one hidden layer, 100 neurons
    X_rp = tf.placeholder(dtype = tf.float32, shape = (None, D), name = 'X_rp')
    Y_rp = tf.placeholder(dtype = tf.int64, shape = (None), name = 'Y_rp')
    X_sp = tf.placeholder(dtype = tf.float32, shape = (None, D), name = 'X_sp')
    Y_sp = tf.placeholder(dtype = tf.int64, shape = (None), name = 'Y_sp')

    # temperature
    if temperature is None:
        T = tf.Variable(tf.random_normal([1]), dtype = tf.float32, name = 'temperature')
        T_constrained = tf.math.sigmoid(T) * 2.8 + 0.2
    else:
        T_constrained = tf.constant(temperature, dtype = tf.float32, name = 'temperature')

    hidden_shared_rp = tf.identity(X_rp)
    hidden_shared_sp = tf.identity(X_sp)
    with tf.name_scope("dnn"):
        # use M_share and M_specific to build DNN
        ################ Build generic layers ################################
        W0 = tf.Variable(tf.random_normal([D, n_hidden]), dtype = tf.float32, name = '0_share_W')
        b0 = tf.Variable(tf.random_normal([n_hidden]), dtype = tf.float32, name = '0_share_bias')
        hidden_shared_rp = tf.nn.relu(tf.add(tf.matmul(X_rp, W0), b0))
        hidden_shared_sp = tf.nn.relu(tf.add(tf.matmul(X_sp, W0), b0))

        for i in range(1,M_share):
            name = str(i)+'_share'
            hidden_shared_rp, hidden_shared_sp = standard_shared_layer(hidden_shared_rp, hidden_shared_sp, n_hidden, name)
        # assign to task specific
        hidden_specific_rp = tf.identity(hidden_shared_rp)
        hidden_specific_sp = tf.identity(hidden_shared_sp)

        ################ Build domain-specific layers ################################
        for i in range(M_specific):
            name = np.str(i)+'_specific'
            hidden_specific_rp,hidden_specific_sp = standard_specific_layer(hidden_specific_rp, hidden_specific_sp, n_hidden, name)
        # output
        output_rp = tf.layers.dense(hidden_specific_rp, K_rp, name = 'output_specific_rp')
        output_sp = tf.div(tf.layers.dense(hidden_specific_sp, K_sp, name = 'output_specific_sp'), T_constrained) # divide SP part by temperature, similar to the scale in nested logit model.
        prob_rp = tf.nn.softmax(output_rp, name = 'prob_rp')
        prob_sp = tf.nn.softmax(output_sp, name = 'prob_sp')

        ####### build regularizers
        vars_ = tf.trainable_variables()
        #
        shared_vars = [var_ for var_ in vars_ if 'share' in var_.name]
        rp_specific_vars = [var_ for var_ in vars_ if 'specific_rp' in var_.name]
        sp_specific_vars = [var_ for var_ in vars_ if 'specific_sp' in var_.name]
        rp_sp_diff_vars = []
        for var_rp, var_sp in zip(rp_specific_vars, sp_specific_vars):
            if 'output' not in var_sp.name:
                rp_sp_diff_vars.append(tf.math.abs(var_sp - var_rp))
            elif 'bias' in var_sp.name:
                rp_sp_diff_vars.append(tf.math.abs(tf.slice(var_sp, [0], [K_rp]) - var_rp))
            else:
                rp_sp_diff_vars.append(tf.math.abs(tf.slice(var_sp, [0,0], [-1, K_rp]) - var_rp))
        # else
        shared_reg = tf.contrib.layers.l1_l2_regularizer(scale_l1=l_shared,scale_l2=l_shared,scope=None)
        rp_sp_diff_reg = tf.contrib.layers.l1_l2_regularizer(scale_l1=l_rp_sp,scale_l2=l_rp_sp,scope=None)
        sp_specific_reg = tf.contrib.layers.l1_l2_regularizer(scale_l1=l_sp,scale_l2=l_sp,scope=None)
        # 
        shared_regularization_penalty=tf.contrib.layers.apply_regularization(shared_reg, shared_vars)
        rp_sp_specific_regularization_penalty=tf.contrib.layers.apply_regularization(rp_sp_diff_reg, rp_sp_diff_vars)
        sp_specific_regularization_penalty=tf.contrib.layers.apply_regularization(sp_specific_reg, sp_specific_vars)

    with tf.name_scope("cost"):
        cost_rp = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output_rp, labels = Y_rp), name = 'cost_rp')
        cost_sp = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output_sp, labels = Y_sp), name = 'cost_sp')
        # adjust the weights between rp and sp costs
        cost_total = tf.add(sp_rp_N_ratio*cost_rp, cost_sp)
        # add reg costs
        cost_total+=shared_regularization_penalty
        cost_total+=rp_sp_specific_regularization_penalty
        cost_total+=sp_specific_regularization_penalty
        
    with tf.name_scope("eval"):
        predict_rp = tf.argmax(output_rp, axis = 1)
        predict_sp = tf.argmax(output_sp, axis = 1)
        correct_rp = tf.nn.in_top_k(output_rp, Y_rp, 1)       
        correct_sp = tf.nn.in_top_k(output_sp, Y_sp, 1)       
        accuracy_rp = tf.reduce_mean(tf.cast(correct_rp, 'float'))
        accuracy_sp = tf.reduce_mean(tf.cast(correct_sp, 'float'))
    
    optimizer = tf.train.AdamOptimizer() # opt objective
    training_op = optimizer.minimize(cost_total) # minimize the opt objective
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    #with tf.Session() as sess:
    # always run this to train the model
    init.run()

    for i in range(n_iterations):
        # print
        if i % 2000 == 0:
            print("---")
            print("")
            print("Iteration", i, "Cost = ", cost_total.eval(feed_dict = {X_rp: X_rp_train, Y_rp: Y_rp_train, X_sp: X_sp_train, Y_sp: Y_sp_train}))
        # gradient descent
        X_rp_batch, Y_rp_batch, X_sp_batch, Y_sp_batch = obtain_mini_batch_rpsp(X_rp_train, Y_rp_train, X_sp_train, Y_sp_train, n_mini_batch)
        sess.run(training_op, feed_dict = {X_rp: X_rp_batch, Y_rp: Y_rp_batch, X_sp: X_sp_batch, Y_sp: Y_sp_batch})
        ''' evaluate the model by testing data'''

    # evaluate
    dict_train = {X_rp: X_rp_train, Y_rp: Y_rp_train, X_sp: X_sp_train, Y_sp: Y_sp_train}
    dict_test = {X_rp: X_rp_test, Y_rp: Y_rp_test, X_sp: X_sp_test, Y_sp: Y_sp_test}

    # cost_total
    train_cost_total = cost_total.eval(feed_dict = dict_train)
    train_cost_rp = cost_rp.eval(feed_dict = dict_train)
    train_cost_sp = cost_sp.eval(feed_dict = dict_train)
    test_cost_total = cost_total.eval(feed_dict = dict_test)
    test_cost_rp = cost_rp.eval(feed_dict = dict_test)
    test_cost_sp = cost_sp.eval(feed_dict = dict_test)

    #
    test_rp_predict = predict_rp.eval(feed_dict = dict_test)
    test_sp_predict = predict_sp.eval(feed_dict = dict_test)
    test_rp_utility = output_rp.eval(feed_dict = dict_test)
    test_sp_utility = output_sp.eval(feed_dict = dict_test)
    test_rp_prob = prob_rp.eval(feed_dict = dict_test)
    test_sp_prob = prob_sp.eval(feed_dict = dict_test)
    test_rp_confusion = confusion_matrix(Y_rp_test, test_rp_predict)
    test_sp_confusion = confusion_matrix(Y_sp_test, test_sp_predict)
    test_rp_accuracy = np.sum(np.diag(test_rp_confusion))/np.sum(test_rp_confusion)
    test_sp_accuracy = np.sum(np.diag(test_sp_confusion))/np.sum(test_sp_confusion)

    #
    train_rp_predict = predict_rp.eval(feed_dict = dict_train)
    train_sp_predict = predict_sp.eval(feed_dict = dict_train)
    train_rp_utility = output_rp.eval(feed_dict = dict_train)
    train_sp_utility = output_sp.eval(feed_dict = dict_train)
    train_rp_prob = prob_rp.eval(feed_dict = dict_train)
    train_sp_prob = prob_sp.eval(feed_dict = dict_train)
    train_rp_confusion = confusion_matrix(Y_rp_train, train_rp_predict)
    train_sp_confusion = confusion_matrix(Y_sp_train, train_sp_predict)
    train_rp_accuracy = np.sum(np.diag(train_rp_confusion))/np.sum(train_rp_confusion)
    train_sp_accuracy = np.sum(np.diag(train_sp_confusion))/np.sum(train_sp_confusion)

    # save results
    test_params = {}
    train_params = {}
    cost = {}
    params = {}

    cost['train_cost_total']=train_cost_total
    cost['train_cost_rp']=train_cost_rp
    cost['train_cost_sp']=train_cost_sp
    cost['test_cost_total']=test_cost_total
    cost['test_cost_rp']=test_cost_rp
    cost['test_cost_sp']=test_cost_sp
    if temperature is None:
        params['T'] = T_constrained.eval()
    else:
        params['T'] = [temperature]

    # test
    test_params['test_rp_predict'] = test_rp_predict
    test_params['test_sp_predict'] = test_sp_predict
    test_params['test_rp_utility'] = test_rp_utility
    test_params['test_sp_utility'] = test_sp_utility
    test_params['test_rp_prob'] = test_rp_prob
    test_params['test_sp_prob'] = test_sp_prob
    test_params['test_rp_confusion'] = test_rp_confusion
    test_params['test_sp_confusion'] = test_sp_confusion
    test_params['test_rp_accuracy'] = test_rp_accuracy
    test_params['test_sp_accuracy'] = test_sp_accuracy
    # train
    train_params['train_rp_predict'] = train_rp_predict
    train_params['train_sp_predict'] = train_sp_predict
    train_params['train_rp_utility'] = train_rp_utility
    train_params['train_sp_utility'] = train_sp_utility
    test_params['train_rp_prob'] = train_rp_prob
    test_params['train_sp_prob'] = train_sp_prob
    train_params['train_rp_confusion'] = train_rp_confusion
    train_params['train_sp_confusion'] = train_sp_confusion
    train_params['train_rp_accuracy'] = train_rp_accuracy
    train_params['train_sp_accuracy'] = train_sp_accuracy

    if (test_rp_accuracy + test_sp_accuracy * 5) / 6 > 0.57:
        train_params['elasticity_train'] = compute_elasticity(X_sp, X_sp_train, [8,9,10,14,15], prob_sp, sess)
        train_params['elasticity_test'] = compute_elasticity(X_sp, X_sp_test, [8,9,10,14,15], prob_sp, sess)
        train_params['choice_prob'] = compute_prob_curve(X_sp, X_sp_train, [8,9,10,14,15], prob_sp, sess)
        ## save models
        saver.save(sess, "models/"+MODEL_NAME+".ckpt")

    sess.close()

    return (test_params,train_params,cost,params)


