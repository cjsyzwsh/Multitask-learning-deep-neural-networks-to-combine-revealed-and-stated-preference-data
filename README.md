# Multitask learning deep neural networks to combine RP and SP data
Work done by Shenhao Wang, Qingyi Wang, and Jinhua Zhao

It is an enduring question how to combine revealed preference (RP) and stated preference (SP) data to analyze individual choices. This study presents a framework of multitask learning deep neural networks (MTLDNNs) to jointly analyze RP and SP, demonstrating its theoretical flexibility and empirical benefits in terms of prediction and economic interpretation. Theoretically, the MTLDNN framework is more generic than the classical nested logit (NL) method because of MTLDNNs’ capacity of automatic feature learning and flexible regularization. Empirically, MTLDNNs outperform six benchmark models and particularly the classical NL models by about 5% prediction accuracy in analyzing the adoption of autonomous vehicles (AVs) based on a survey data collected in Singapore. This improvement can be mainly attributed to the soft constraints specific to multitask learning, including its innovative architectural design and regularization methods, but not much to the generic capacity of automatic feature learning endowed by a standard feedforward DNN architecture. Besides prediction, MTLDNNs are also interpretable. MTLDNNs reveal that AVs mainly substitute driving and that the variables specific to AVs are more important than the socio-economic variables in determining AV adoption. Overall, this study presents a new MTLDNN framework in combining RP and SP, demonstrates its theoretical flexibility of architectural design and regularizations, shows its empirical predictive power, and extracts reasonable economic information for interpretation. Future studies can explore other MTLDNN architectures, new choice modeling applications for multitask learning, and deeper theoretical connections between the MTLDNN framework and structural choice modeling.

Folders:
 - dnn: Code for pooled and separated dnns used as a baseline.
 - nl: Code for nested logit model used as a baseline
 
 The scripts under code are multitask learning DNNs proposed in the paper.
 
 - 0_clean_data.py: cleans and preprocesses data
 - 1_hyper_training_dnn_rp_sp.py: hyper parameter searching 
 - 2_mtldnn_prediction_result_plot.py: plot prediction accuracy w.r.t. different hyperparameters (shared layers, specific layers, etc.)
 - 3_load_model_analyze.py: compute and graph elasticity and choice probability
 - utils.py: utility functions for constructing and analyze neural networks
 
 Models are named and loaded based on the hyperparameters in the model. parames_to_run.csv and params_to_load.csv are sample files used for hyperparameter searching and loading models for analysis.
 
#### Note: codes and paper are included in this repository. Unfortunately, we cannot upload the data set due to the policy limitation.
