import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import pickle as pkl
import utils

f = open('../../data/processed_data/SGP.pickle', "rb")
data = pkl.load(f)
f.close()
data_sp = data["X_train_sp"]
data_sp["rp"] = 0
data_sp["sp"] = 1
choice_sp = data["Y_train_sp"]
data_sp['choice'] = choice_sp + 1

data_test_sp = data["X_test_sp"]
data_test_sp["rp"] = 0
data_test_sp["sp"] = 1
choice_test_sp = data["Y_test_sp"]
data_test_sp['choice'] = choice_test_sp + 1

database = db.Database("SGP", data_sp)
database_test = db.Database("SGP_test", data_test_sp)

from headers import *
  
#Parameters to be estimated
# Arguments:
#   1  Name for report. Typically, the same as the variable
#   2  Starting value
#   3  Lower bound
#   4  Upper bound
#   5  0: estimate the parameter, 1: keep it fixed

#### SP Betas ####
ASC_SP_DRIVE	 = Beta('ASC_SP_DRIVE',0,-10,10,1)
ASC_SP_WALK	 = Beta('ASC_SP_WALK',0,-20,20,0)
ASC_SP_BUS	 = Beta('ASC_SP_BUS',0,-20,20,0)
ASC_SP_RS	 = Beta('ASC_SP_RS',0,-20,20,0)
ASC_SP_AV	 = Beta('ASC_SP_AV',0,-20,20,0)


B_SP_AGE_DRIVE	 = Beta('B_SP_AGE_DRIVE',0,-10,10,0)
B_SP_INC_DRIVE	 = Beta('B_SP_INC_DRIVE',0,-10,10,0)
B_SP_EDU_DRIVE	 = Beta('B_SP_EDU_DRIVE',0,-10,10,0)
B_SP_MALE_DRIVE	 = Beta('B_SP_MALE_DRIVE',0,-10,10,0)
B_SP_YOUNG_DRIVE	 = Beta('B_SP_YOUNG_DRIVE',0,-10,10,0)
B_SP_OLD_DRIVE	 = Beta('B_SP_OLD_DRIVE',0,-10,10,0)
B_SP_LEDU_DRIVE	 = Beta('B_SP_LEDU_DRIVE',0,-10,10,0)
B_SP_HEDU_DRIVE	 = Beta('B_SP_HEDU_DRIVE',0,-10,10,0)
B_SP_FULLJOB_SP_DRIVE	 = Beta('B_SP_FULLJOB_SP_DRIVE',0,-10,10,0)

B_SP_COST_DRIVE	 = Beta('B_SP_COST_DRIVE',0,-10,10,0)
B_SP_WALKTIME_DRIVE	 = Beta('B_SP_WALKTIME_DRIVE',0,-10,10,0)
B_SP_IVT_DRIVE	 = Beta('B_SP_IVT_DRIVE',0,-10,10,0)


B_SP_AGE_BUS	 = Beta('B_SP_AGE_BUS',0,-10,10,0)
B_SP_INC_BUS	 = Beta('B_SP_INC_BUS',0,-10,10,0)
B_SP_EDU_BUS	 = Beta('B_SP_EDU_BUS',0,-10,10,0)
B_SP_MALE_BUS	 = Beta('B_SP_MALE_BUS',0,-10,10,0)
B_SP_YOUNG_BUS	 = Beta('B_SP_YOUNG_BUS',0,-10,10,0)
B_SP_OLD_BUS	 = Beta('B_SP_OLD_BUS',0,-10,10,0)
B_SP_LEDU_BUS	 = Beta('B_SP_LEDU_BUS',0,-10,10,0)
B_SP_HEDU_BUS	 = Beta('B_SP_HEDU_BUS',0,-10,10,0)
B_SP_FULLJOB_SP_BUS	 = Beta('B_SP_FULLJOB_SP_BUS',0,-10,10,0)

B_SP_COST_BUS	 = Beta('B_SP_COST_BUS',0,-10,10,0)
B_SP_WALKTIME_BUS	 = Beta('B_SP_WALKTIME_BUS',0,-10,10,0)
B_SP_IVT_BUS	 = Beta('B_SP_IVT_BUS',0,-10,10,0)
B_SP_WAITTIME_BUS = Beta('B_SP_WAITTIME_BUS', 0, -10, 10, 0)


B_SP_AGE_WALK	 = Beta('B_SP_AGE_WALK',0,-10,10,0)
B_SP_INC_WALK	 = Beta('B_SP_INC_WALK',0,-10,10,0)
B_SP_EDU_WALK	 = Beta('B_SP_EDU_WALK',0,-10,10,0)
B_SP_MALE_WALK	 = Beta('B_SP_MALE_WALK',0,-10,10,0)
B_SP_YOUNG_WALK	 = Beta('B_SP_YOUNG_WALK',0,-10,10,0)
B_SP_OLD_WALK	 = Beta('B_SP_OLD_WALK',0,-10,10,0)
B_SP_LEDU_WALK	 = Beta('B_SP_LEDU_WALK',0,-10,10,0)
B_SP_HEDU_WALK	 = Beta('B_SP_HEDU_WALK',0,-10,10,0)
B_SP_FULLJOB_SP_WALK	 = Beta('B_SP_FULLJOB_SP_WALK',0,-10,10,0)

B_SP_WALKTIME_WALK	 = Beta('B_SP_WALKTIME_WALK',0,-10,10,0)

B_SP_AGE_RS	 = Beta('B_SP_AGE_RS',0,-10,10,0)
B_SP_INC_RS	 = Beta('B_SP_INC_RS',0,-10,10,0)
B_SP_EDU_RS	 = Beta('B_SP_EDU_RS',0,-10,10,0)
B_SP_MALE_RS	 = Beta('B_SP_MALE_RS',0,-10,10,0)
B_SP_YOUNG_RS	 = Beta('B_SP_YOUNG_RS',0,-10,10,0)
B_SP_OLD_RS	 = Beta('B_SP_OLD_RS',0,-10,10,0)
B_SP_LEDU_RS	 = Beta('B_SP_LEDU_RS',0,-10,10,0)
B_SP_HEDU_RS	 = Beta('B_SP_HEDU_RS',0,-10,10,0)
B_SP_FULLJOB_SP_RS	 = Beta('B_SP_FULLJOB_SP_RS',0,-10,10,0)

B_SP_COST_RS	 = Beta('B_SP_COST_RS',0,-10,10,0)
B_SP_IVT_RS	 = Beta('B_SP_IVT_RS',0,-10,10,0)
B_SP_WAITTIME_RS = Beta('B_SP_WAITTIME_RS', 0, -10, 10, 0)

B_SP_AGE_AV	 = Beta('B_SP_AGE_AV',0,-10,10,0)
B_SP_INC_AV	 = Beta('B_SP_INC_AV',0,-10,10,0)
B_SP_EDU_AV	 = Beta('B_SP_EDU_AV',0,-10,10,0)
B_SP_MALE_AV	 = Beta('B_SP_MALE_AV',0,-10,10,0)
B_SP_YOUNG_AV	 = Beta('B_SP_YOUNG_AV',0,-10,10,0)
B_SP_OLD_AV	 = Beta('B_SP_OLD_AV',0,-10,10,0)
B_SP_LEDU_AV	 = Beta('B_SP_LEDU_AV',0,-10,10,0)
B_SP_HEDU_AV	 = Beta('B_SP_HEDU_AV',0,-10,10,0)
B_SP_FULLJOB_SP_AV	 = Beta('B_SP_FULLJOB_SP_AV',0,-10,10,0)

B_SP_COST_AV	 = Beta('B_SP_COST_AV',0,-10,10,0)
B_SP_IVT_AV	 = Beta('B_SP_IVT_AV',0,-10,10,0)
B_SP_WAITTIME_AV = Beta('B_SP_WAITTIME_AV', 0, -10, 10, 0)

# Define here arithmetic expressions for name that are not directly 
# available from the data
one  = DefineVariable('one',1, database)
SP_AV = DefineVariable('SP_AV', sp, database)

# Utilities              
DRIVE_SP = ASC_SP_DRIVE + B_SP_AGE_DRIVE * age + B_SP_INC_DRIVE * inc + B_SP_EDU_DRIVE * edu \
        + B_SP_MALE_DRIVE * male + B_SP_YOUNG_DRIVE * young_age + B_SP_OLD_DRIVE * old_age \
        + B_SP_LEDU_DRIVE * low_edu + B_SP_HEDU_DRIVE * high_edu + B_SP_FULLJOB_SP_DRIVE * full_job \
        + B_SP_COST_DRIVE * drive_cost + B_SP_WALKTIME_DRIVE * drive_walktime \
        + B_SP_IVT_DRIVE * drive_ivt

WALK_SP = ASC_SP_WALK + B_SP_AGE_WALK * age + B_SP_INC_WALK * inc + B_SP_EDU_WALK * edu \
        + B_SP_MALE_WALK * male + B_SP_YOUNG_WALK * young_age + B_SP_OLD_WALK * old_age \
        + B_SP_LEDU_WALK * low_edu + B_SP_HEDU_WALK * high_edu + B_SP_FULLJOB_SP_WALK * full_job \
        + B_SP_WALKTIME_WALK * walk_walktime
        
BUS_SP = ASC_SP_BUS + B_SP_AGE_BUS * age + B_SP_INC_BUS * inc + B_SP_EDU_BUS * edu \
        + B_SP_MALE_BUS * male + B_SP_YOUNG_BUS * young_age + B_SP_OLD_BUS * old_age \
        + B_SP_LEDU_BUS * low_edu + B_SP_HEDU_BUS * high_edu + B_SP_FULLJOB_SP_BUS * full_job \
        + B_SP_COST_BUS * bus_cost + B_SP_WALKTIME_BUS * bus_walktime \
        + B_SP_IVT_BUS * bus_ivt + B_SP_WAITTIME_BUS * bus_waittime
        
RS_SP = ASC_SP_RS + B_SP_AGE_RS * age + B_SP_INC_RS * inc + B_SP_EDU_RS * edu \
        + B_SP_MALE_RS * male + B_SP_YOUNG_RS * young_age + B_SP_OLD_RS * old_age \
        + B_SP_LEDU_RS * low_edu + B_SP_HEDU_RS * high_edu + B_SP_FULLJOB_SP_RS * full_job \
        + B_SP_COST_RS * ridesharing_cost + B_SP_IVT_RS * ridesharing_ivt \
        + B_SP_WAITTIME_RS * ridesharing_waittime
        
AV_SP = ASC_SP_AV + B_SP_AGE_AV * age + B_SP_INC_AV * inc + B_SP_EDU_AV * edu \
        + B_SP_MALE_AV * male + B_SP_YOUNG_AV * young_age + B_SP_OLD_AV * old_age \
        + B_SP_LEDU_AV * low_edu + B_SP_HEDU_AV * high_edu + B_SP_FULLJOB_SP_AV * full_job \
        + B_SP_COST_AV * av_cost + B_SP_IVT_AV * av_ivt \
        + B_SP_WAITTIME_AV * av_waittime

V = {1:WALK_SP, 2:BUS_SP, 3:RS_SP, 4:DRIVE_SP, 5:AV_SP}
av = {1:sp, 2:sp, 3:sp, 4:sp, 5:sp}

# Estimation of the model
logprob = bioLogLogit(V,av,choice)
logprob1 = bioLogLogit(V,av,1)
logprob2 = bioLogLogit(V,av,2)
logprob3 = bioLogLogit(V,av,3)
logprob4 = bioLogLogit(V,av,4)
logprob5 = bioLogLogit(V,av,5)

biogeme  = bio.BIOGEME(database,logprob)
biogeme.modelName = "MNL_SP"
results = biogeme.estimate()

# Print the estimated values
betas = results.getBetaValues()
for k,v in betas.items():
    print(f"{k}=\t{v:.3g}")

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)
pandasResults.to_csv("mnl_sp_coefs.csv")

# Training Accuracy
simulate = {'logprob1' : logprob1, 'logprob2' : logprob2, 'logprob3' : logprob3, \
            'logprob4' : logprob4, 'logprob5' : logprob5}

joint_acc, rp_acc, sp_acc, chosen = utils.get_accuracy(simulate, database, betas, data_sp)
    
data_sp.to_csv("data_mnl_sp.csv", index=False)
    
print(joint_acc, rp_acc, sp_acc)

# Testing Accuracy
joint_acc, rp_acc, sp_acc, chosen = utils.get_accuracy(simulate, database_test, betas, data_test_sp)

data_test_sp.to_csv("data_mnl_sp_test.csv", index=False)

print(joint_acc, rp_acc, sp_acc)
