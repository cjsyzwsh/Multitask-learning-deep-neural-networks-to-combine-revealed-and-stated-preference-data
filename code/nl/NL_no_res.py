import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import pickle as pkl
import utils

data, data_test, choice, choice_test = utils.load_data()

database = db.Database("SGP", data)
database_test = db.Database("SGP_test", data_test)

from headers import *
  
#Parameters to be estimated
# Arguments:
#   1  Name for report. Typically, the same as the variable
#   2  Starting value
#   3  Lower bound
#   4  Upper bound
#   5  0: estimate the parameter, 1: keep it fixed

ASC_RP_DRIVE	 = Beta('ASC_RP_DRIVE',0,-20,20,1)
ASC_RP_WALK	 = Beta('ASC_RP_WALK',0,-10,10,0)
ASC_RP_BUS	 = Beta('ASC_RP_BUS',0,-20,20,0)
ASC_RP_RS	 = Beta('ASC_RP_RS',0,-20,20,0)
ASC_SP_DRIVE	 = Beta('ASC_SP_DRIVE',0,-10,10,0)
ASC_SP_WALK	 = Beta('ASC_SP_WALK',0,-20,20,0)
ASC_SP_BUS	 = Beta('ASC_SP_BUS',0,-20,20,0)
ASC_SP_RS	 = Beta('ASC_SP_RS',0,-20,20,0)
ASC_SP_AV	 = Beta('ASC_SP_AV',0,-20,20,0)

#### RP Betas ####
B_RP_AGE_DRIVE	 = Beta('B_RP_AGE_DRIVE',0,-10,10,0)
B_RP_INC_DRIVE	 = Beta('B_RP_INC_DRIVE',0,-10,10,0)
B_RP_EDU_DRIVE	 = Beta('B_RP_EDU_DRIVE',0,-10,10,0)
B_RP_MALE_DRIVE	 = Beta('B_RP_MALE_DRIVE',0,-10,10,0)
B_RP_YOUNG_DRIVE	 = Beta('B_RP_YOUNG_DRIVE',0,-10,10,0)
B_RP_OLD_DRIVE	 = Beta('B_RP_OLD_DRIVE',0,-10,10,0)
B_RP_LEDU_DRIVE	 = Beta('B_RP_LEDU_DRIVE',0,-10,10,0)
B_RP_HEDU_DRIVE	 = Beta('B_RP_HEDU_DRIVE',0,-10,10,0)
B_RP_FULLJOB_RP_DRIVE	 = Beta('B_RP_FULLJOB_RP_DRIVE',0,-10,10,0)

B_RP_COST_DRIVE	 = Beta('B_RP_COST_DRIVE',0,-10,10,0)
B_RP_WALKTIME_DRIVE	 = Beta('B_RP_WALKTIME_DRIVE',0,-10,10,0)
B_RP_IVT_DRIVE	 = Beta('B_RP_IVT_DRIVE',0,-10,10,0)


B_RP_AGE_BUS	 = Beta('B_RP_AGE_BUS',0,-10,10,0)
B_RP_INC_BUS	 = Beta('B_RP_INC_BUS',0,-10,10,0)
B_RP_EDU_BUS	 = Beta('B_RP_EDU_BUS',0,-10,10,0)
B_RP_MALE_BUS	 = Beta('B_RP_MALE_BUS',0,-10,10,0)
B_RP_YOUNG_BUS	 = Beta('B_RP_YOUNG_BUS',0,-10,10,0)
B_RP_OLD_BUS	 = Beta('B_RP_OLD_BUS',0,-10,10,0)
B_RP_LEDU_BUS	 = Beta('B_RP_LEDU_BUS',0,-10,10,0)
B_RP_HEDU_BUS	 = Beta('B_RP_HEDU_BUS',0,-10,10,0)
B_RP_FULLJOB_RP_BUS	 = Beta('B_RP_FULLJOB_RP_BUS',0,-10,10,0)

B_RP_COST_BUS	 = Beta('B_RP_COST_BUS',0,-10,10,0)
B_RP_WALKTIME_BUS	 = Beta('B_RP_WALKTIME_BUS',0,-10,10,0)
B_RP_IVT_BUS	 = Beta('B_RP_IVT_BUS',0,-10,10,0)
B_RP_WAITTIME_BUS = Beta('B_RP_WAITTIME_BUS', 0, -10, 10, 0)


B_RP_AGE_WALK	 = Beta('B_RP_AGE_WALK',0,-10,10,0)
B_RP_INC_WALK	 = Beta('B_RP_INC_WALK',0,-10,10,0)
B_RP_EDU_WALK	 = Beta('B_RP_EDU_WALK',0,-10,10,0)
B_RP_MALE_WALK	 = Beta('B_RP_MALE_WALK',0,-10,10,0)
B_RP_YOUNG_WALK	 = Beta('B_RP_YOUNG_WALK',0,-10,10,0)
B_RP_OLD_WALK	 = Beta('B_RP_OLD_WALK',0,-10,10,0)
B_RP_LEDU_WALK	 = Beta('B_RP_LEDU_WALK',0,-10,10,0)
B_RP_HEDU_WALK	 = Beta('B_RP_HEDU_WALK',0,-10,10,0)
B_RP_FULLJOB_RP_WALK	 = Beta('B_RP_FULLJOB_RP_WALK',0,-10,10,0)

B_RP_WALKTIME_WALK	 = Beta('B_RP_WALKTIME_WALK',0,-10,10,0)

B_RP_AGE_RS	 = Beta('B_RP_AGE_RS',0,-10,10,0)
B_RP_INC_RS	 = Beta('B_RP_INC_RS',0,-10,10,0)
B_RP_EDU_RS	 = Beta('B_RP_EDU_RS',0,-10,10,0)
B_RP_MALE_RS	 = Beta('B_RP_MALE_RS',0,-10,10,0)
B_RP_YOUNG_RS	 = Beta('B_RP_YOUNG_RS',0,-10,10,0)
B_RP_OLD_RS	 = Beta('B_RP_OLD_RS',0,-10,10,0)
B_RP_LEDU_RS	 = Beta('B_RP_LEDU_RS',0,-10,10,0)
B_RP_HEDU_RS	 = Beta('B_RP_HEDU_RS',0,-10,10,0)
B_RP_FULLJOB_RP_RS	 = Beta('B_RP_FULLJOB_RP_RS',0,-10,10,0)

B_RP_COST_RS	 = Beta('B_RP_COST_RS',0,-10,10,0)
B_RP_IVT_RS	 = Beta('B_RP_IVT_RS',0,-10,10,0)
B_RP_WAITTIME_RS = Beta('B_RP_WAITTIME_RS', 0, -10, 10, 0)

#### SP Betas ####

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

# parameters relevant to the nests
MU_SP = Beta('MU_SP',1,0,10,0)

# Define here arithmetic expressions for name that are not directly 
# available from the data
one  = DefineVariable('one',1, database)

# Utilities
DRIVE_RP = ASC_RP_DRIVE + B_RP_AGE_DRIVE * age + B_RP_INC_DRIVE * inc + B_RP_EDU_DRIVE * edu \
        + B_RP_MALE_DRIVE * male + B_RP_YOUNG_DRIVE * young_age + B_RP_OLD_DRIVE * old_age \
        + B_RP_LEDU_DRIVE * low_edu + B_RP_HEDU_DRIVE * high_edu + B_RP_FULLJOB_RP_DRIVE * full_job \
        + B_RP_COST_DRIVE * drive_cost + B_RP_WALKTIME_DRIVE * drive_walktime \
        + B_RP_IVT_DRIVE * drive_ivt


WALK_RP = ASC_RP_WALK + B_RP_AGE_WALK * age + B_RP_INC_WALK * inc + B_RP_EDU_WALK * edu \
        + B_RP_MALE_WALK * male + B_RP_YOUNG_WALK * young_age + B_RP_OLD_WALK * old_age \
        + B_RP_LEDU_WALK * low_edu + B_RP_HEDU_WALK * high_edu + B_RP_FULLJOB_RP_WALK * full_job \
        + B_RP_WALKTIME_WALK * walk_walktime
        
BUS_RP = ASC_RP_BUS + B_RP_AGE_BUS * age + B_RP_INC_BUS * inc + B_RP_EDU_BUS * edu \
        + B_RP_MALE_BUS * male + B_RP_YOUNG_BUS * young_age + B_RP_OLD_BUS * old_age \
        + B_RP_LEDU_BUS * low_edu + B_RP_HEDU_BUS * high_edu + B_RP_FULLJOB_RP_BUS * full_job \
        + B_RP_COST_BUS * bus_cost + B_RP_WALKTIME_BUS * bus_walktime \
        + B_RP_IVT_BUS * bus_ivt + B_RP_WAITTIME_BUS * bus_waittime
         
RS_RP = ASC_RP_RS + B_RP_AGE_RS * age + B_RP_INC_RS * inc + B_RP_EDU_RS * edu \
        + B_RP_MALE_RS * male + B_RP_YOUNG_RS * young_age + B_RP_OLD_RS * old_age \
        + B_RP_LEDU_RS * low_edu + B_RP_HEDU_RS * high_edu + B_RP_FULLJOB_RP_RS * full_job \
        + B_RP_COST_RS * ridesharing_cost + B_RP_IVT_RS * ridesharing_ivt \
        + B_RP_WAITTIME_RS * ridesharing_waittime
              
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

V = {1:WALK_RP, 2:BUS_RP, 3:RS_RP, 4:DRIVE_RP, 5:WALK_SP, 6:BUS_SP, 7:RS_SP, 8:DRIVE_SP, 9:AV_SP}
av = {1:rp, 2:rp, 3:rp, 4:rp, 5:sp, 6:sp, 7:sp, 8:sp, 9:sp}

# Definition of nests
rp = 1.0, [1,2,3,4]
sp = MU_SP, [5,6,7,8,9]

nests = rp, sp

# Estimation of the model
logprob = models.lognested(V,av,nests,choice)
logprob1 = models.lognested(V,av,nests,1)
logprob2 = models.lognested(V,av,nests,2)
logprob3 = models.lognested(V,av,nests,3)
logprob4 = models.lognested(V,av,nests,4)
logprob5 = models.lognested(V,av,nests,5)
logprob6 = models.lognested(V,av,nests,6)
logprob7 = models.lognested(V,av,nests,7)
logprob8 = models.lognested(V,av,nests,8) 
logprob9 = models.lognested(V,av,nests,9)


biogeme  = bio.BIOGEME(database,logprob)
biogeme.modelName = "NL_NO_RES"
results = biogeme.estimate()

# Print the estimated values
betas = results.getBetaValues()
for k,v in betas.items():
    print(f"{k}=\t{v:.3g}")

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)
pandasResults.to_csv("nl_no_res_coefs.csv")

# Training Accuracy
simulate = {'logprob1' : logprob1, 'logprob2' : logprob2, 'logprob3' : logprob3, \
            'logprob4' : logprob4, 'logprob5' : logprob5, 'logprob6' : logprob6, \
            'logprob7' : logprob7, 'logprob8' : logprob8, 'logprob9' : logprob9}

joint_acc, rp_acc, sp_acc, chosen = utils.get_accuracy(simulate, database, betas, data)
    
data.to_csv("data_nl_no_res.csv", index=False)
    
print(joint_acc, rp_acc, sp_acc)

# Testing Accuracy
joint_acc, rp_acc, sp_acc, chosen = utils.get_accuracy(simulate, database_test, betas, data_test)

data_test.to_csv("data_nl_no_res_test.csv", index=False)

print(joint_acc, rp_acc, sp_acc)
