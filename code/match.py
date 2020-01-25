#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:06:55 2019

@author: qingyi
"""
import pandas as pd
import numpy as np

if __name__ == "__main__":
     
    to_run = pd.read_csv("Round_3_0708/results_prev.csv")
    torun_params = to_run[['# shared layers','# specific layers','# hidden units','l_shared','l_rp_sp','l_sp']]
    torun_params = torun_params.round(20)
    torun_params = [tuple(x) for x in torun_params.values]
    
    temp = pd.read_csv("Round_3_0708/params_to_run.csv")
    temp = temp.round(20)
    temp = [tuple(x) for x in temp.values]
    torun_params = torun_params + temp

    ran = pd.read_csv("Round_2_07/results.csv")
    ran = ran.round(20)
    ran_params = ran[['# shared layers','# specific layers','# hidden units','l_shared','l_rp_sp','l_sp']]
    results_prev = []
    params_torun = []
    ran_params = [tuple(x) for x in ran_params.values]
    for x in torun_params:
        if tuple(x) in ran_params:
            results_prev.append(ran.iloc[ran_params.index(tuple(x))])
        else:
            params_torun.append(tuple(x))

    results_prev = pd.DataFrame(results_prev, columns=['# shared layers','# specific layers','# hidden units','l_shared','l_rp_sp','l_sp', \
                                   'tr_rp_accuracy','tr_sp_accuracy','tr_overall_accuracy','rp_accuracy','sp_accuracy', \
                                   'overall_accuracy','temperature'])
    results_prev.to_csv("results_prev.csv", index=False)
    #pd.DataFrame(params_torun, columns  = ['# shared layers','# specific layers','# hidden units','l_shared','l_rp_sp','l_sp']).to_csv("params_to_run.csv", index=False)
