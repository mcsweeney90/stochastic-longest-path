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
summary_path = "../summaries/empirical_dists"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "../plots/empirical_dists"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

nb = 128
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]
distros = ["NORMAL", "GAMMA", "UNIFORM"]
# Load info.   
with open('../data/chol_empirical.dill', 'rb') as file:
    chol_empirical = dill.load(file)    
with open('../data/stg_empirical.dill', 'rb') as file:
    stg_empirical = dill.load(file)   

# =============================================================================
# Summary statistics for Cholesky set.
# =============================================================================

# with open("{}/chol_summary_statistics.txt".format(summary_path), "w") as dest:
#     print("TESTING THE NORMALITY ASSUMPTION OF THE EMPIRICAL DISTRIBUTION FOR THE CHOLESKY DAG SET.", file=dest) 
#     print("EMPIRICAL DISTRIBUTIONS GENERATED THROUGH 100,000 REALIZATIONS OF ENTIRE GRAPH.", file=dest)    
#     for nt in n_tasks:
#         print("\n\n\n---------------------------------", file=dest)
#         print("NUMBER OF TASKS: {}".format(nt), file=dest)
#         print("---------------------------------", file=dest)
#         for dist in distros: 
#             print("\n{} WEIGHTS".format(dist), file=dest)
#             mu = np.mean(chol_empirical[nt][dist])
#             print("MEAN: {}".format(mu), file=dest)
#             var = np.var(chol_empirical[nt][dist])
#             print("VARIANCE/STD: {} / {}".format(var, np.sqrt(var)), file=dest)
#             skw = skew(chol_empirical[nt][dist])
#             print("SKEWNESS: {}".format(skw), file=dest)
#             kur = kurtosis(chol_empirical[nt][dist])
#             print("EXCESS KURTOSIS: {}".format(kur), file=dest)
#             # stat, p = shapiro(chol_empirical[nt][dist])
#             # verdict = "NORMAL" if p > 0.05 else "NOT NORMAL"
#             # print("p = {} - {}".format(p, verdict), file=dest)
#             med = np.median(chol_empirical[nt][dist])
#             print("MEDIAN: {}".format(med), file=dest)
#             mx = max(chol_empirical[nt][dist])
#             print("MAXIMUM: {}".format(mx), file=dest)
#             mn = min(chol_empirical[nt][dist])
#             print("MINIMUM: {}".format(mn), file=dest)

# =============================================================================
# Progression of MC solutions for Cholesky set.
# =============================================================================

# with open("{}/chol_mc.txt".format(summary_path), "w") as dest:
#     print("QUALITY OF MOMENT ESTIMATES FOR MONTE CARLO METHOD WITH INCREASING NUMBERS OF SAMPLES.", file=dest) 
#     print("REFERENCE SOLUTION COMPUTED WITH 100,000 SAMPLES.", file=dest)    
#     for nt in n_tasks:
#         print("\n\n\n---------------------------------", file=dest)
#         print("NUMBER OF TASKS: {}".format(nt), file=dest)
#         print("---------------------------------", file=dest)
#         for dist in distros: 
#             print("\n{} WEIGHTS".format(dist), file=dest)
#             mu = np.mean(chol_empirical[nt][dist])
#             print("REFERENCE MEAN: {}".format(mu), file=dest)
#             for s in [10, 100, 1000, 10000]:
#                 m = np.mean(chol_empirical[nt][dist][:s])
#                 diff = 100 - (m/mu)*100
#                 print("{} SAMPLES (value, %diff) : ({}, {})".format(s, m, diff), file=dest)
#             print("---------------------------------", file=dest)
#             var = np.var(chol_empirical[nt][dist])
#             print("REFERENCE VARIANCE: {}".format(var), file=dest)
#             for s in [10, 100, 1000, 10000]:
#                 v = np.var(chol_empirical[nt][dist][:s])
#                 diff = 100 - (v/var)*100
#                 print("{} SAMPLES (value, %diff) : ({}, {})".format(s, v, diff), file=dest)            
            
# =============================================================================
# Histograms of empirical distributions for Cholesky set.
# =============================================================================

# for dist in distros:
#     fig = plt.figure(dpi=400) 
#     ax = fig.add_subplot(111, frameon=False)
#     # hide tick and tick label of the big axes
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     w = len(n_tasks) / 2
#     for i, nt in enumerate(n_tasks):
#         ax1 = fig.add_subplot(w*100 + 21 + i)
#         ax1.hist(chol_empirical[nt][dist], color='#348ABD', density=True, bins='auto', align='mid')
#         plt.axvline(np.mean(chol_empirical[nt][dist]), color='#E24A33', linestyle='solid', linewidth=1, label='MEAN')
#         ax1.xaxis.grid(False)
#         ax1.yaxis.grid(False)         
#         lft = min(chol_empirical[nt][dist]) - 5
#         rt = max(chol_empirical[nt][dist]) + 5
#         ax1.set_xlim(left=lft, right=rt)
#         if i == 0:
#             ax1.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white')        
#         plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#         ax1.set_title("n = {}".format(nt), weight='bold')    
#     plt.subplots_adjust(wspace=0)        
#     plt.savefig('{}/histograms_{}'.format(plot_path, dist), bbox_inches='tight') 
#     plt.close(fig) 

# =============================================================================
# Progression of variance for Cholesky set with normal weights. Not used anywhere.
# =============================================================================            

# diffs = defaultdict(list)
# for nt in n_tasks:
#     var = np.var(chol_empirical[nt]["NORMAL"])
#     for s in [10, 100, 1000, 10000]:        
#         v = np.var(chol_empirical[nt]["NORMAL"][:s])
#         d = 100 - (v/var)*100
#         diffs[s].append(d)
# colors = {10 : '#E24A33', 100 : '#348ABD', 1000 : '#988ED5', 10000 : '#FBC15E'}

# fig = plt.figure(dpi=400)
# ax1 = fig.add_subplot(111)
# for s in [10, 100, 1000, 10000]:
#     ax1.plot(n_tasks, diffs[s], color=colors[s], label=s)
# # plt.yscale('log')
# ax1.set_xlabel("DAG SIZE", labelpad=5)
# ax1.set_ylabel("VARIANCE DEVIATION (%)", labelpad=5)
# ax1.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white', title='#SAMPLES') 
# plt.savefig('{}/mc_variance'.format(plot_path), bbox_inches='tight') 
# plt.close(fig) 

# =============================================================================
# Comparison of STG distributions.
# =============================================================================

# devs = {}
# for dist in ["NORMAL", "UNIFORM"]:
#     devs[dist] = {}
#     devs[dist]["MEAN"] = []
#     devs[dist]["VAR"] = []
#     for dname in stg_empirical:
#         dmu = abs(100 - (stg_empirical[dname][dist][0] / stg_empirical[dname]["GAMMA"][0])*100)
#         devs[dist]["MEAN"].append(dmu)    
#         dvar = abs(100 - (stg_empirical[dname][dist][1] / stg_empirical[dname]["GAMMA"][1])*100)
#         devs[dist]["VAR"].append(dvar)    

# with open("{}/stg_comparison.txt".format(summary_path), "w") as dest:
#     print("COMPARISON OF MOMENT ESTIMATES FOR STG GRAPH EMPIRICAL DISTRIBUTIONS COMPUTED VIA MC METHOD WITH 10,000 SAMPLES.", file=dest)  
#     print("TRUE SOLUTION ASSUMED TO THAT COMPUTED WITH ALL GAMMA WEIGHTS.", file=dest) 
#     print("HOW LARGE ARE THE MOMENT DEVIATIONS FOR MC SOLUTIONS WITH NORMAL OR UNIFORM WEIGHTS?", file=dest)
    
#     print("\n\n\n---------------------------------", file=dest)
#     print("MEAN DEVIATIONS (%)", file=dest)
#     print("---------------------------------", file=dest)    
#     for dist in ["NORMAL", "UNIFORM"]:
#         print("\n{} WEIGHTS".format(dist), file=dest)
#         print("MAXIMUM : {}".format(max(devs[dist]["MEAN"])), file=dest)
#         print("AVERAGE : {}".format(np.mean(devs[dist]["MEAN"])), file=dest)
#         gr1 = sum(1 for m in devs[dist]["MEAN"] if m > 1.0)
#         gr5 = sum(1 for m in devs[dist]["MEAN"] if m > 5.0)
#         gr10 = sum(1 for m in devs[dist]["MEAN"] if m > 10.0)
#         print("#TIMES > 1% : {} / 1620".format(gr1), file=dest)
#         print("#TIMES > 5% : {} / 1620".format(gr5), file=dest)
#         print("#TIMES > 10% : {} / 1620".format(gr10), file=dest)
    
#     print("\n\n\n---------------------------------", file=dest)
#     print("VARIANCE DEVIATIONS (%)", file=dest)
#     print("---------------------------------", file=dest) 
#     for dist in ["NORMAL", "UNIFORM"]:
#         print("\n{} WEIGHTS".format(dist), file=dest)
#         print("MAXIMUM : {}".format(max(devs[dist]["VAR"])), file=dest)
#         print("AVERAGE : {}".format(np.mean(devs[dist]["VAR"])), file=dest)
#         gr1 = sum(1 for m in devs[dist]["VAR"] if m > 1.0)
#         gr5 = sum(1 for m in devs[dist]["VAR"] if m > 5.0)
#         gr10 = sum(1 for m in devs[dist]["VAR"] if m > 10.0)
#         print("#TIMES > 1% : {} / 1620".format(gr1), file=dest)
#         print("#TIMES > 5% : {} / 1620".format(gr5), file=dest)
#         print("#TIMES > 10% : {} / 1620".format(gr10), file=dest)
    
    
    
    