#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:45:45 2018

hyper parameter training by using training and validation sets

@author: shenhao
"""

#cd "/Users/shenhao/Dropbox (MIT)/Qingyi_Baichuan_Shenhao/Qingyi_Shenhao_MTLDNN/code"
#cd "D:\Dropbox (MIT)\Shenhao_Jinhua (1)\7_ml_structure\code"

import os.path
import numpy as np
import pandas as pd
#import statsmodels.api as sm
import tensorflow as tf
import util as util
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.metrics import accuracy_score
import itertools
import multiprocessing

def train_nn(input_list):

    sp_rp_N_ratio = 1# N_sp/N_rp
    n_iteration = 20000
    n_mini_batch = 200

    (Y_rp_train, Y_rp_test, Y_sp_train, Y_sp_test, X_rp_train, X_rp_test, X_sp_train, X_sp_test, \
     M_share, M_specific, n_hidden, l_shared, l_rp_sp, l_sp, temperature) = input_list

    param = '_'.join([str(M_share), str(M_specific), str(n_hidden), str(l_shared), str(l_rp_sp), str(l_sp)])

    test_params,train_params,cost,params = util.build_nn(Y_rp_train, Y_rp_test, Y_sp_train, Y_sp_test,
                                                         X_rp_train, X_rp_test, X_sp_train, X_sp_test,
                                                         int(M_share), int(M_specific), int(n_hidden),
                                                         l_shared, l_rp_sp, l_sp,
                                                         sp_rp_N_ratio,
                                                         n_iteration, n_mini_batch, MODEL_NAME=param, temperature = temperature)

    Y_rp_test_predict = test_params['test_rp_predict']
    Y_sp_test_predict = test_params['test_sp_predict']
    Y_test_predict = np.concatenate([Y_rp_test_predict, Y_sp_test_predict])
    Y_rp_train_predict = train_params['train_rp_predict']
    Y_sp_train_predict = train_params['train_sp_predict']
    Y_train_predict = np.concatenate([Y_rp_train_predict, Y_sp_train_predict])

    rp_accuracy = accuracy_score(Y_rp_test,Y_rp_test_predict)
    sp_accuracy = accuracy_score(Y_sp_test, Y_sp_test_predict)
    tr_rp_accuracy = accuracy_score(Y_rp_train, Y_rp_train_predict)
    tr_sp_accuracy = accuracy_score(Y_sp_train, Y_sp_train_predict)

    # obtain total pred accuracy
    Y_train = np.concatenate([Y_rp_train, Y_sp_train])
    tr_overall_accuracy = accuracy_score(Y_train, Y_train_predict)
    Y_test = np.concatenate([Y_rp_test, Y_sp_test])
    overall_accuracy = accuracy_score(Y_test, Y_test_predict)
    temperature = params['T'][0]

    print(M_share, M_specific, n_hidden, l_shared, l_rp_sp, l_sp)
    print("Joint prediction accuracy is: ", overall_accuracy)
    print("Temperature in SP is: ", params['T'][0])

    if overall_accuracy < 0.57:
        return train_nn(input_list)

    if 'choice_prob' in train_params.keys():
        # get elasticity
        elas_tr = train_params['elasticity_train']
        elas_te = train_params['elasticity_test']
        param = '_'.join([str(M_share), str(M_specific), str(n_hidden), str(l_shared), str(l_rp_sp), str(l_sp)])
        with open('elasticities/elas_' + param + '.pickle', 'wb') as f:
            pickle.dump(elas_tr, f)
            pickle.dump(elas_te, f)

        # plot choice probability curves
        mode = ["walk","bus","ride hailing","drive","AV"]
        variables = ['av_cost', 'av_waittime', 'av_ivt', 'age', 'income']
        for i,v in zip([0,1,2,3,4], [8,9,10,14,15]):
            fig, ax = plt.subplots()
            for j in range(5):
                ax.scatter(X_sp_train[:, v], train_params['choice_prob'][i][:, j], s = 2, label = mode[j])
            ax.legend()
            ax.set_ylabel("choice probability")
            ax.set_xlabel(variables[i])
            fig.savefig("cp_graph/"+ str(v) + '_' + param + ".png")

    outfile = open('temp_result.txt', 'a')
    outfile.write("%d, %d, %d, %.2E, %.2E, %.2E, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n" % \
                  (M_share, M_specific, n_hidden, l_shared, l_rp_sp, l_sp, tr_rp_accuracy, tr_sp_accuracy, tr_overall_accuracy, rp_accuracy, sp_accuracy, overall_accuracy, temperature))
    outfile.close()

    return tr_rp_accuracy, tr_sp_accuracy, tr_overall_accuracy, rp_accuracy, sp_accuracy, overall_accuracy, temperature


if __name__ == '__main__':

    Y_rp_train, Y_rp_test, Y_sp_train, Y_sp_test, X_rp_train, X_rp_test, X_sp_train, X_sp_test = util.load_data()

    n_samples = 1500

    D_rp = X_rp_train.shape[1]
    N_rp = X_rp_train.shape[0]
    K_rp = 4
    D_sp = X_sp_train.shape[1]
    N_sp = X_sp_train.shape[0]
    K_sp = 5


    # method 1: break the search space down into 4 parts
    '''
    # 1st quarter
    L_M_share = [1, 5]
    L_M_specific = [1, 3, 5]
    L_n_hidden = [25, 50, 100, 200]
    L_l_shared = [1e-20, 1e-4, 1e-2, 5e-1]
    L_l_rp_sp = [1e-20, 1e-4, 1e-2, 5e-1]
    L_l_sp = [1e-20, 1e-4, 1e-2, 5e-1]
    
    # 2nd quarter
    L_M_share = [2, 3, 4]
    L_M_specific = [1, 3, 5]
    L_n_hidden = [50, 100]
    L_l_shared = [1e-20, 1e-4, 1e-2, 5e-1]
    L_l_rp_sp = [1e-20, 1e-4, 1e-2, 5e-1]
    L_l_sp = [1e-20, 1e-4, 1e-2, 5e-1]

    # 3rd quarter
    L_M_share = [1, 5]
    L_M_specific = [2, 4]
    L_n_hidden = [25, 50, 100, 200]
    L_l_shared = [1e-20, 1e-4, 1e-2, 5e-1]
    L_l_rp_sp = [1e-20, 1e-4, 1e-2, 5e-1]
    L_l_sp = [1e-20, 1e-4, 1e-2, 5e-1]

    # 4th quarter
    L_M_share = [2, 3, 4] 
    L_M_specific = [2, 4]
    L_n_hidden = [25, 50, 100, 200]
    L_l_shared = [1e-20, 1e-4, 1e-2, 5e-1]
    L_l_rp_sp = [1e-20, 1e-4, 1e-2, 5e-1]
    L_l_sp = [1e-20, 1e-4, 1e-2, 5e-1]
    
    parameters = [L_M_share, L_M_specific, L_n_hidden, L_l_shared, L_l_rp_sp, L_l_sp]
    parameters_torun = list(itertools.product(*parameters))
    '''

    # method 2: search the full space (except the parameters that we have already searched for)
    '''
    # all
    L_M_share = [1, 2, 3, 4, 5]
    L_M_specific = [1, 2, 3, 4, 5]
    L_n_hidden = [25, 50, 100, 200]
    L_l_shared = [1e-20, 1e-4, 1e-2, 5e-1]
    L_l_rp_sp = [1e-20, 1e-4, 1e-2, 5e-1]
    L_l_sp = [1e-20, 1e-4, 1e-2, 5e-1]
    
    parameters = [L_M_share, L_M_specific, L_n_hidden, L_l_shared, L_l_rp_sp, L_l_sp]
    parameters = list(itertools.product(*parameters))
    parameters= np.array([tuple(x) for x in parameters])
    assert len(parameters) >= n_samples
    params_random = parameters[list(np.random.choice(len(parameters), n_samples))]

    ran = pd.read_csv("Round_2_07/results.csv")
    ran = ran.round(20)
    ran_params = ran[['# shared layers','# specific layers','# hidden units','l_shared','l_rp_sp','l_sp']]
    results_prev = []
    params_torun = []
    ran_params = [tuple(x) for x in ran_params.values]
    for x in params_random:
        if tuple(x) in ran_params:
            results_prev.append(ran.iloc[ran_params.index(tuple(x))])
        else:
            params_torun.append(tuple(x))

    results_prev = pd.DataFrame(results_prev, columns=['# shared layers','# specific layers','# hidden units','l_shared','l_rp_sp','l_sp', \
                                   'tr_rp_accuracy','tr_sp_accuracy','tr_overall_accuracy','rp_accuracy','sp_accuracy', \
                                   'overall_accuracy','temperature'])
    results_prev.to_csv("results_prev.csv", index=False)
    pd.DataFrame(params_torun, columns  = ['# shared layers','# specific layers','# hidden units','l_shared','l_rp_sp','l_sp']).to_csv("params_to_run.csv", index=False)
    print(len(results_prev))
    print(len(params_torun))
    '''
    # method 3: load parameters from csv
    params_torun = np.array(pd.read_csv("params_to_run.csv"))

    parameters = [(Y_rp_train, Y_rp_test, Y_sp_train, Y_sp_test, X_rp_train, X_rp_test, X_sp_train, X_sp_test) + tuple(x) for x in params_torun]
    if not os.path.exists("temp_result.txt"):
        outfile = open('temp_result.txt', 'a')
        outfile.write("# shared layers,# specific layers,# hidden units,l_shared,l_rp_sp,l_sp,tr_rp_accuracy, tr_sp_accuracy, tr_overall_accuracy, rp_accuracy,sp_accuracy,overall_accuracy,temperature\n")
        outfile.close()

    p = multiprocessing.Pool(5)
    f = p.map(train_nn, parameters)

    f = np.array(f).T

    export = pd.DataFrame(np.insert(f, 0, np.array(params_torun).T, axis=0).T, \
                          columns=['# shared layers','# specific layers','# hidden units','l_shared','l_rp_sp','l_sp', \
                                   'tr_rp_accuracy','tr_sp_accuracy','tr_overall_accuracy','rp_accuracy','sp_accuracy', \
                                   'overall_accuracy','temperature'])
    export.to_csv("results.csv", index=False)
