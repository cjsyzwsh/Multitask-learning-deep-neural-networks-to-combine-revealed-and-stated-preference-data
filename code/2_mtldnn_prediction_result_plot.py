# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 21:15:29 2019

@author: wangqi44
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 24})

run_dir = "Round_3_0708/"
results = pd.read_csv(run_dir + "results.csv")
results = results.sort_values(by='overall_accuracy', ascending=False)
high_acc = int(len(results) * 0.1)

results['temperature'] = (results['temperature'] - 0.2) // 0.3 * 0.3 + 0.2

# Accuracy vs. all hyperparameters
fig1, ax1 = plt.subplots(figsize = (10, 7))
ax1.scatter(results['# shared layers'][:high_acc], results['overall_accuracy'][:high_acc], color = 'r', label = '')
fig2, ax2 = plt.subplots(figsize = (10, 7))
ax2.scatter(results['# specific layers'][:high_acc], results['overall_accuracy'][:high_acc], color = 'r', label = '')
fig3, ax3 = plt.subplots(figsize = (10, 7))
ax3.scatter(np.log10(results['l_rp_sp'][:high_acc]), results['overall_accuracy'][:high_acc], color = 'r', label = '')
fig4, ax4 = plt.subplots(figsize = (10, 7))
ax4.scatter(np.log10(results['l_sp'][:high_acc]), results['overall_accuracy'][:high_acc], color = 'r', label = '')
fig5, ax5 = plt.subplots(figsize = (10, 7))
ax5.scatter(np.log10(results['l_shared'][:high_acc]), results['overall_accuracy'][:high_acc], color = 'r', label = '')
fig6, ax6 = plt.subplots(figsize = (10, 7))
ax6.scatter(results['# hidden units'][:high_acc], results['overall_accuracy'][:high_acc], color = 'r', label = '')
fig7, ax7 = plt.subplots(figsize = (10, 7))
ax7.scatter(results['temperature'][:high_acc], results['overall_accuracy'][:high_acc], color = 'r', label = '')

maxi = results[:high_acc].groupby("# shared layers", as_index=False).max()[['# shared layers', 'overall_accuracy']]
ax1.plot(maxi['# shared layers'], maxi['overall_accuracy'], color = 'r', linewidth = 2, label = 'top 10% accuracy runs')
maxi = results[:high_acc].groupby("# specific layers", as_index=False).max()[['# specific layers', 'overall_accuracy']]
ax2.plot(maxi['# specific layers'], maxi['overall_accuracy'], color = 'r', linewidth = 2, label = 'top 10% accuracy runs')
maxi = results[:high_acc].groupby("l_rp_sp", as_index=False).max()[['l_rp_sp', 'overall_accuracy']]
ax3.plot(np.log10(maxi['l_rp_sp']), maxi['overall_accuracy'], color = 'r', linewidth = 2, label = 'top 10% accuracy runs')
maxi = results[:high_acc].groupby("l_sp", as_index=False).max()[['l_sp', 'overall_accuracy']]
ax4.plot(np.log10(maxi['l_sp']), maxi['overall_accuracy'], color = 'r', linewidth = 2, label = 'top 10% accuracy runs')
maxi = results[:high_acc].groupby("l_shared", as_index=False).max()[['l_shared', 'overall_accuracy']]
ax5.plot(np.log10(maxi['l_shared']), maxi['overall_accuracy'], color = 'r', linewidth = 2, label = 'top 10% accuracy runs')
maxi = results[:high_acc].groupby("# hidden units", as_index=False).max()[['# hidden units', 'overall_accuracy']]
ax6.plot(maxi['# hidden units'], maxi['overall_accuracy'], color = 'r', linewidth = 2, label = 'top 10% accuracy runs')
maxi = results[:high_acc].groupby("temperature", as_index=False).max()[['temperature', 'overall_accuracy']]
ax7.plot(maxi['temperature'], maxi['overall_accuracy'], color = 'r', linewidth = 2, label = 'top 10% accuracy runs')

ax1.scatter(results['# shared layers'][high_acc:], results['overall_accuracy'][high_acc:], color = 'b', label = '')
ax2.scatter(results['# specific layers'][high_acc:], results['overall_accuracy'][high_acc:], color = 'b', label = '')
ax3.scatter(np.log10(results['l_rp_sp'][high_acc:]), results['overall_accuracy'][high_acc:], color = 'b', label = '')
ax4.scatter(np.log10(results['l_sp'][high_acc:]), results['overall_accuracy'][high_acc:], color = 'b', label = '')
ax5.scatter(np.log10(results['l_shared'][high_acc:]), results['overall_accuracy'][high_acc:], color = 'b', label = '')
ax6.scatter(results['# hidden units'][high_acc:], results['overall_accuracy'][high_acc:], color = 'b', label = '')
ax7.scatter(results['temperature'][high_acc:], results['overall_accuracy'][high_acc:], color = 'b', label = '')

mean = results.groupby("# shared layers", as_index=False).mean()[['# shared layers', 'overall_accuracy']]
ax1.plot(mean['# shared layers'], mean['overall_accuracy'], color = 'b', linewidth = 2, label = 'all runs')
mean = results.groupby("# specific layers", as_index=False).mean()[['# specific layers', 'overall_accuracy']]
ax2.plot(mean['# specific layers'], mean['overall_accuracy'], color = 'b', linewidth = 2, label = 'all runs')
mean = results.groupby("l_rp_sp", as_index=False).mean()[['l_rp_sp', 'overall_accuracy']]
ax3.plot(np.log10(mean['l_rp_sp']), mean['overall_accuracy'], color = 'b', linewidth = 2, label = 'all runs')
mean = results.groupby("l_sp", as_index=False).mean()[['l_sp', 'overall_accuracy']]
ax4.plot(np.log10(mean['l_sp']), mean['overall_accuracy'], color = 'b', linewidth = 2, label = 'all runs')
mean = results.groupby("l_shared", as_index=False).mean()[['l_shared', 'overall_accuracy']]
ax5.plot(np.log10(mean['l_shared']), mean['overall_accuracy'], color = 'b', linewidth = 2, label = 'all runs')
mean = results.groupby("# hidden units", as_index=False).mean()[['# hidden units', 'overall_accuracy']]
ax6.plot(mean['# hidden units'], mean['overall_accuracy'], color = 'b', linewidth = 2, label = 'all runs')
mean = results.groupby("temperature", as_index=False).mean()[['temperature', 'overall_accuracy']]
ax7.plot(mean['temperature'], mean['overall_accuracy'], color = 'b', linewidth = 2, label = 'all runs')

ax1.set_xlabel("# shared layers")
ax2.set_xlabel("# specific layers")
ax3.set_xlabel("log(l_rp_sp)")
ax4.set_xlabel("log(l_sp)")
ax5.set_xlabel("log(l_shared)")
ax6.set_xlabel("# hidden units")
ax7.set_xlabel("temperature")

ax1.set_ylabel("overall accuracy")
ax2.set_ylabel("overall accuracy")
ax3.set_ylabel("overall accuracy")
ax4.set_ylabel("overall accuracy")
ax5.set_ylabel("overall accuracy")
ax6.set_ylabel("overall accuracy")
ax7.set_ylabel("overall accuracy")

ax1.set_ylim([0.4,0.65])
ax2.set_ylim([0.4,0.65])
ax3.set_ylim([0.4,0.65])
ax4.set_ylim([0.4,0.65])
ax5.set_ylim([0.4,0.65])
ax6.set_ylim([0.4,0.65])
ax7.set_ylim([0.4,0.65])

ax1.legend(fancybox=True, framealpha=0.5)
ax2.legend(fancybox=True, framealpha=0.5)
ax3.legend(fancybox=True, framealpha=0.5)
ax4.legend(fancybox=True, framealpha=0.5)
ax5.legend(fancybox=True, framealpha=0.5)
ax6.legend(fancybox=True, framealpha=0.5)
ax7.legend(fancybox=True, framealpha=0.5)

'''fig1.savefig(run_dir + "sharedlayers.png")
fig2.savefig(run_dir + "specificlayers.png")
fig3.savefig(run_dir + "l_rp_sp.png")
fig4.savefig(run_dir + "l_sp.png")
fig5.savefig(run_dir + "l_shared.png")
fig6.savefig(run_dir + "hiddenunits.png")
fig7.savefig(run_dir + "temperature.png")'''

# Accuracy vs. Architecture
filt = results[results['# shared layers'] + results['# specific layers'] == 5]

fig9, ax9 = plt.subplots(figsize = (10, 7))
temp = pd.read_csv("dnn/temp_result_sep_nn.csv")
temp["# shared layers"] = 0
filt = filt.append(temp[['# shared layers', 'overall_accuracy']])
temp = pd.read_csv("dnn/temp_result_pooled_nn.csv")
temp["# shared layers"] = 5
filt = filt.append(temp[['# shared layers', 'overall_accuracy']])
filt = filt.sort_values(by='overall_accuracy', ascending=False)
high_acc = int(len(filt) * 0.1)

ax9.scatter(filt['# shared layers'][high_acc:], filt['overall_accuracy'][high_acc:], color = 'b', label = 'all runs')
ax9.scatter(filt['# shared layers'][:high_acc], filt['overall_accuracy'][:high_acc], color = 'r', label = 'top 10% accuracy runs')
maxi = filt.groupby("# shared layers", as_index=False).max()[['# shared layers', 'overall_accuracy']]
mean = filt.groupby("# shared layers", as_index=False).mean()[['# shared layers', 'overall_accuracy']]
#maxi = maxi.append(pd.DataFrame([[5, 0.527]], columns = ['# shared layers', 'overall_accuracy']))
#maxi = pd.DataFrame([[0, 0.532]], columns = ['# shared layers', 'overall_accuracy']).append(maxi)
ax9.plot(mean['# shared layers'], mean['overall_accuracy'], color = 'b', label = '')
ax9.plot(maxi['# shared layers'], maxi['overall_accuracy'], color = 'r', label = '')
ax9.set_ylim([0.4,0.65])
ax9.legend(fancybox=True, framealpha=0.5)
ax9.set_xlabel("# shared layers") 
#fig9.savefig(run_dir + "architecture.png")
