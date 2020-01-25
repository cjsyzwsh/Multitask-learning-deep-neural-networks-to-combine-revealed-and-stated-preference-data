# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:25:47 2019

@author: wangqi44
"""

import numpy as np
import pandas as pd
import pickle as pkl
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import seaborn as sns

def load_data():
    f = open('../../data/processed_data/SGP.pickle', "rb")
    data = pkl.load(f)
    f.close()
    # Training Data
    data_rp = data["X_train_rp"]
    data_sp = data["X_train_sp"]
    data_rp["rp"] = 1
    data_sp["rp"] = 0
    data_rp["sp"] = 0
    data_sp["sp"] = 1
    choice_rp = data["Y_train_rp"]
    choice_sp = data["Y_train_sp"]
    choice_sp = np.max(choice_rp) + 1 + choice_sp
    choice = np.concatenate((choice_rp, choice_sp))
    
    # Testing Data
    data_test_rp = data["X_test_rp"]
    data_test_sp = data["X_test_sp"]
    data_test_rp["rp"] = 1
    data_test_sp["rp"] = 0
    data_test_rp["sp"] = 0
    data_test_sp["sp"] = 1
    choice_test_rp = data["Y_test_rp"]
    choice_test_sp = data["Y_test_sp"]
    choice_test_sp = np.max(choice_test_rp) + 1 + choice_test_sp
    choice_test = np.concatenate((choice_test_rp, choice_test_sp))
    
    # Pool Data Together
    data = pd.concat([data_rp, data_sp])
    data_test = pd.concat([data_test_rp, data_test_sp])
    
    data['choice'] = choice + 1
    data_test['choice'] = choice_test + 1

    return data, data_test, choice, choice_test

def get_accuracy(simulate, database, betas, data):
    biogeme2  = bio.BIOGEME(database,simulate)
    simulatedValues = biogeme2.simulate(betas)
    
    prob = np.zeros((len(data),np.max(data['choice'])))
    cumulative = np.zeros(len(data))
    i = 0
    for col in simulatedValues.columns:
        cumulative += np.exp(np.array(simulatedValues[col]))
        prob[:, i] = cumulative
        i += 1
    choose = np.random.rand(len(data))
    
    chosen_draw = []
    for c, bins in zip(choose, prob):
        chosen_draw.append(np.digitize(c, bins))
    
    chosen = np.argmax(np.array(simulatedValues), axis = 1) 
    prob = np.exp(np.max(np.array(simulatedValues), axis = 1))
    sns.distplot(prob[(chosen_draw != chosen) & (chosen+1 == data['choice'])], kde=False)
    data['predicted_choice'] = chosen
    data['predicted_choice'] += 1
    
    joint_acc = np.sum(data['predicted_choice'] == data['choice']) / len(data)
    rp_acc = np.sum(data[data['rp'] == 1]['predicted_choice'] == data[data['rp'] == 1]['choice']) / len(data[data['rp'] == 1])
    sp_acc = np.sum(data[data['rp'] == 0]['predicted_choice'] == data[data['rp'] == 0]['choice']) / len(data[data['rp'] == 0])
    
    return joint_acc, rp_acc, sp_acc, chosen

