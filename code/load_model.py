import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import util

plt.rcParams.update({'font.size': 24})
output_folder = '190827'
mode = ["walk","bus","ride hailing","drive","AV"]
variables = ['av_cost ($)', 'av_waittime (min)', 'av_ivt (min)', 'age', 'monthly income ($1000)']
params = pd.read_csv("params_to_load.csv")
params = np.array(params)

colors = ['salmon', 'wheat','darkseagreen','plum','dodgerblue']
axes = []
for i in range(5):
    axes.append(plt.subplots(figsize=(10,7)))
    axes[i][1].set_ylabel("choice probability")
    axes[i][1].set_xlabel(variables[i])


with open('../data/processed_data/SGP_raw.pickle', 'rb') as f:
    SGP = pickle.load(f)
# sp
X_sp_train_raw = SGP['X_train_sp'].values    
X_rp_train_raw = SGP['X_train_rp'].values
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
var_names = ['Walk time',
         'Public transit cost', 'Public transit walk time', 'Public transit wait time', 'Public transit in-vehicle time',
         'Ride hail cost', 'Ride hail wait time', 'Ride hail in-vehicle time',
         'AV cost', 'AV wait time', 'AV in-vehicle time',
         'Drive cost', 'Drive walk time', 'Drive in-vehicle time',
         'Age', 'Income', 'edu',
         'male','young_age', 'old_age',
         'low_edu', 'high_edu',
         'low_inc', 'high_inc',
         'full_job']
X_sp_train_raw = pd.DataFrame(X_sp_train_raw, columns = names)
X_rp_train_raw = pd.DataFrame(X_rp_train_raw, columns = names)

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
X_sp_train_standard = X_sp_train_raw[standard_vars]
X_sp_train_nonstandard = X_sp_train_raw[non_standard_vars]
X_rp_train_standard = X_rp_train_raw[standard_vars]
X_rp_train_nonstandard = X_rp_train_raw[non_standard_vars]

elasvars_sp = [0,1,2,3,4,5,6,7,11,12,13,8,9,10,14,15]
elasvars_rp = [0,1,2,3,4,5,6,7,11,12,13,14,15]
std_sp = np.sqrt(StandardScaler().fit(X_sp_train_standard).var_[elasvars_sp])
std_rp = np.sqrt(StandardScaler().fit(X_rp_train_standard).var_[elasvars_rp])
#StandardScaler().fit(X_sp_train_standard).mean_[[8,9,10,14,15]]

X_sp_train_raw = np.array(X_sp_train_raw)
X_rp_train_raw = np.array(X_rp_train_raw)

Y_rp_train, Y_rp_test, Y_sp_train, Y_sp_test, X_rp_train, X_rp_test, X_sp_train, X_sp_test = util.load_data()
for i in range(min(2,len(params))):
    (M_share, M_specific, n_hidden, l_shared, l_rp_sp, l_sp, __) = params[i]
    
    param = '_'.join([str(M_share), str(M_specific), str(n_hidden), str(l_shared), str(l_rp_sp), str(l_sp)])
    tf.reset_default_graph() 

    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph("../output/" + output_folder + "/models/"+param+".ckpt.meta")
    saver.restore(sess, "../output/" + output_folder + "/models/"+ param+".ckpt")
    
    graph = tf.get_default_graph()
    X_sp = graph.get_tensor_by_name("X_sp:0")
    Y_sp = graph.get_tensor_by_name("Y_sp:0")
    X_rp = graph.get_tensor_by_name("X_rp:0")
    Y_rp = graph.get_tensor_by_name("Y_rp:0")

    prob_rp = graph.get_tensor_by_name("dnn/prob_rp:0")
    prob_sp = graph.get_tensor_by_name("dnn/prob_sp:0")
    
    predict_rp = tf.argmax(prob_rp, axis = 1)
    predict_sp = tf.argmax(prob_sp, axis = 1)
    #print(confusion_matrix(Y_sp_train, sess.run(predict_sp, feed_dict={X_sp: X_sp_train})))
    
    prob_sp_train = sess.run(prob_sp, feed_dict={X_sp: X_sp_train})
    prob_sp_test = sess.run(prob_sp, feed_dict={X_sp: X_sp_test})
    prob_rp_train = sess.run(prob_rp, feed_dict={X_rp: X_rp_train})
    prob_rp_test = sess.run(prob_rp, feed_dict={X_rp: X_rp_test})

    # SP elasticity
    e = []
    e_std = []

    grad_sp = []
    for j in [0,1,2,3,4]:
        grad_sp.append(tf.gradients(prob_sp[:, j], X_sp))
        grad = sess.run(grad_sp[-1], feed_dict={X_sp: X_sp_train})[0][:, elasvars_sp]
        elas = grad / prob_sp_train[:, j][:, None] * np.array(X_sp_train_standard)[:, elasvars_sp] / std_sp[None, :]
        e.append(np.mean(elas, axis = 0))
        e_std.append(np.std(elas, axis = 0))
    e = np.array(e).T
    e_std = np.array(e_std).T
    for e_row, e_std_row, v, m1 in zip(e, e_std, elasvars_sp,[0,1,1,1,1,2,2,2,3,3,3,4,4,4,-1,-1]):
        print(var_names[v], end=' & ')
        for print_e, print_e_std, m2 in zip(e_row, e_std_row, range(len(e_row))):
            if print_e == e_row[-1]:
                end_char = "\\\\ \n"
            else:
                end_char = " & "
            if m1 != m2:
                print("%.3f(%.1f)" % (print_e, print_e_std), end = end_char)
            else:
                print("\\textbf{%.3f(%.1f)}" % (print_e, print_e_std), end = end_char)

    # RP elasticity
    '''
    grad_rp = []
    e = []
    e_std = []
    for j in [0,1,2,3]:
        grad_rp.append(tf.gradients(prob_rp[:, j], X_rp))
        grad = sess.run(grad_rp[-1], feed_dict={X_rp: X_rp_train})[0][:, elasvars_rp]
        elas = grad / prob_rp_train[:, j][:, None] * np.array(X_rp_train_standard)[:, elasvars_rp] / std_rp[None, :]
        e.append(np.mean(elas, axis = 0))
        e_std.append(np.std(elas, axis = 0))
    e = np.array(e).T
    e_std = np.array(e_std).T
    for e_row, e_std_row, v, m1 in zip(e, e_std, elasvars_rp,[0,1,1,1,1,2,2,2,3,3,3,-1,-1]):
        print(var_names[v], end=' & ')
        for print_e, print_e_std, m2 in zip(e_row, e_std_row, range(len(e_row))):
            if print_e == e_row[-1]:
                end_char = "\\\\ \n"
            else:
                end_char = " & "
            if m1 != m2:
                print("%.3f(%.1f)" % (print_e, print_e_std), end = end_char)
            else:
                print("\\textbf{%.3f(%.1f)}" % (print_e, print_e_std), end = end_char)
    '''

    # choice prob curves
    average = np.zeros((5, len(Y_sp_train), 5))
    choice_prob = util.compute_prob_curve(X_sp, X_sp_train, [8,9,10,14,15], prob_sp, sess)
    average = average + np.array(choice_prob)

    # plot choice probability curves
    for i,v in zip([0,1,2,3,4], [8,9,10,14,15]):
        for j in range(5):
            axes[i][1].scatter(X_sp_train_raw[:, v], choice_prob[i][:, j], s = 2, alpha = 0.3, color = colors[j])

    sess.close()


'''
colors = ['red','orange','green','purple','blue']
average = average / len(params)
for i,v in zip([0,1,2,3,4], [8,9,10,14,15]):
    limit_max = 0
    limit_min = 100
    for j in range(5):
        tuples = sorted(zip(X_sp_train_raw[:, v], average[i][:,j]))
        x_sorted = [t[0] for t in tuples]
        limit_max = max(x_sorted[int(len(x_sorted)*0.99)], limit_max)
        limit_min = min(x_sorted[0], limit_min)
        axes[i][1].plot([t[0] for t in tuples], [t[1] for t in tuples], linewidth = 4, label = mode[j], color = colors[j])
    axes[i][1].legend(fancybox=True, framealpha=0.3)
    axes[i][1].set_xlim([limit_min, limit_max])
    axes[i][1].set_ylim([0, 1])
    axes[i][0].savefig("cp_graph/"+ str(v) + "_all.png")
'''