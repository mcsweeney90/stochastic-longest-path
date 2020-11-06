#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summaries and plots for the empirical distributions.
"""

import dill, pathlib, sys
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import shapiro, skew, kurtosis

####################################################################################################

# Set some parameters for plots.
# See here: http://www.futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold' 
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.titlepad'] = 1
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.

####################################################################################################

# Destinations to save summaries and generated plots.
summary_path = "summaries/empirical_dists"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/empirical_dists"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

nb = 128
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]
# Load info.
with open('empirical_dists.dill', 'rb') as file:
    empirical = dill.load(file)     

# =============================================================================
# Readable summaries of tests of distribution.
# =============================================================================

# with open("{}/summary_statistics.txt".format(summary_path), "w") as dest:
#     print("TESTING THE NORMALITY ASSUMPTION OF THE EMPIRICAL DISTRIBUTION FOR THE CHOLESKY DAG SET.", file=dest) 
#     print("EMPIRICAL DISTRIBUTIONS GENERATED THROUGH 100,000 REALIZATIONS OF ENTIRE GRAPH.", file=dest)
    
#     for nt in n_tasks:
#         print("\n\n\n---------------------------------", file=dest)
#         print("NUMBER OF TASKS: {}".format(nt), file=dest)
#         print("---------------------------------", file=dest)
#         for dist in ["NORMAL", "GAMMA"]: 
#             print("\n{} WEIGHTS".format(dist), file=dest)
#             mu = np.mean(empirical[nt][dist])
#             print("MEAN: {}".format(mu), file=dest)
#             var = np.var(empirical[nt][dist])
#             print("VARIANCE/STD: {} / {}".format(var, np.sqrt(var)), file=dest)
#             skw = skew(empirical[nt][dist])
#             print("SKEWNESS: {}".format(skw), file=dest)
#             kur = kurtosis(empirical[nt][dist])
#             print("EXCESS KURTOSIS: {}".format(kur), file=dest)
#             # stat, p = shapiro(empirical[nt][dist])
#             # verdict = "NORMAL" if p > 0.05 else "NOT NORMAL"
#             # print("p = {} - {}".format(p, verdict), file=dest)
#             med = np.median(empirical[nt][dist])
#             print("MEDIAN: {}".format(med), file=dest)
#             mx = max(empirical[nt][dist])
#             print("MAXIMUM: {}".format(mx), file=dest)
#             mn = min(empirical[nt][dist])
#             print("MINIMUM: {}".format(mn), file=dest)

# with open("{}/samples.txt".format(summary_path), "w") as dest:
#     print("QUALITY OF MOMENT ESTIMATES FOR MONTE CARLO METHOD WITH INCREASING NUMBERS OF SAMPLES.", file=dest) 
#     print("REFERENCE SOLUTION COMPUTED WITH 100,000 SAMPLES.", file=dest)
    
#     for nt in n_tasks:
#         print("\n\n\n---------------------------------", file=dest)
#         print("NUMBER OF TASKS: {}".format(nt), file=dest)
#         print("---------------------------------", file=dest)
#         for dist in ["NORMAL", "GAMMA"]: 
#             print("\n{} WEIGHTS".format(dist), file=dest)
#             mu = np.mean(empirical[nt][dist])
#             print("REFERENCE MEAN: {}".format(mu), file=dest)
#             for s in [10, 100, 1000, 10000]:
#                 m = np.mean(empirical[nt][dist][:s])
#                 diff = 100 - (m/mu)*100
#                 print("{} SAMPLES (value, %diff) : ({}, {})".format(s, m, diff), file=dest)
#             print("---------------------------------", file=dest)
#             var = np.var(empirical[nt][dist])
#             print("REFERENCE VARIANCE: {}".format(var), file=dest)
#             for s in [10, 100, 1000, 10000]:
#                 v = np.var(empirical[nt][dist][:s])
#                 diff = 100 - (v/var)*100
#                 print("{} SAMPLES (value, %diff) : ({}, {})".format(s, v, diff), file=dest)
            
            
# =============================================================================
# Plots.
# =============================================================================


# Histograms of distributions.
# for dist in ["NORMAL", "GAMMA"]:
#     fig = plt.figure(dpi=400) 
#     ax = fig.add_subplot(111, frameon=False)
#     # hide tick and tick label of the big axes
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     w = len(n_tasks) / 2
#     for i, nt in enumerate(n_tasks):
#         ax1 = fig.add_subplot(w*100 + 21 + i)
#         ax1.hist(empirical[nt][dist], color='#348ABD', density=True, bins='auto', align='mid')
#         plt.axvline(np.mean(empirical[nt][dist]), color='#E24A33', linestyle='solid', linewidth=1, label='MEAN')
#         ax1.xaxis.grid(False)
#         ax1.yaxis.grid(False)         
#         lft = min(empirical[nt][dist]) - 5
#         rt = max(empirical[nt][dist]) + 5
#         ax1.set_xlim(left=lft, right=rt)
#         if i == 0:
#             ax1.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white')        
#         plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#         ax1.set_title("n = {}".format(nt), weight='bold')    
#     plt.subplots_adjust(wspace=0)        
#     plt.savefig('{}/histograms_{}'.format(plot_path, dist), bbox_inches='tight') 
#     plt.close(fig) 
            

# Variance.
diffs = defaultdict(list)
for nt in n_tasks:
    var = np.var(empirical[nt]["NORMAL"])
    for s in [10, 100, 1000, 10000]:        
        v = np.var(empirical[nt]["NORMAL"][:s])
        d = 100 - (v/var)*100
        diffs[s].append(d)
colors = {10 : '#E24A33', 100 : '#348ABD', 1000 : '#988ED5', 10000 : '#FBC15E'}

fig = plt.figure(dpi=400)
ax1 = fig.add_subplot(111)
for s in [10, 100, 1000, 10000]:
    ax1.plot(n_tasks, diffs[s], color=colors[s], label=s)
# plt.yscale('log')
ax1.set_xlabel("DAG SIZE", labelpad=5)
ax1.set_ylabel("VARIANCE DEVIATION (%)", labelpad=5)
ax1.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white', title='#SAMPLES') 
plt.savefig('{}/mc_variance'.format(plot_path), bbox_inches='tight') 
plt.close(fig) 

            