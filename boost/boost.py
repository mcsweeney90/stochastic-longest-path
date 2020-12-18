#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path heuristics analysis.
TODO: add bootstrap confidence intervals?
"""

import dill, pathlib, sys
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, kstest
sys.path.append('../../') # Needed for src apparently...

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
plt.rcParams['axes.titlepad'] = 0
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

####################################################################################################

# Destinations to save summaries and generated plots.
summary_path = "../summaries/mc_boost"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "../plots/mc_boost"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

# Cholesky.
nb = 128
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]
with open('../data/chol_boost.dill', 'rb') as file:
    chol_boost = dill.load(file) 
with open('../data/chol_boost_timings.dill', 'rb') as file:
    chol_boost_timings = dill.load(file) 
with open('../data/chol_existing.dill', 'rb') as file:
    chol_existing = dill.load(file) 
with open('../data/chol_empirical.dill', 'rb') as file:
    chol_empirical = dill.load(file) 
    
# STG.
with open('../data/stg_boost.dill', 'rb') as file:
    stg_boost = dill.load(file) 
with open('../data/stg_existing.dill', 'rb') as file:
    stg_existing = dill.load(file) 
with open('../data/stg_empirical.dill', 'rb') as file:
    stg_empirical = dill.load(file) 

# =============================================================================
# STG set.
# =============================================================================

# datapoints = ["MC30", "MC30-SCULLI", "MC30-CorLCA", "MC30-MC", "MC100", "MC100-SCULLI", "MC100-CorLCA", "MC100-MC"]

# mean_devs = {p : [] for p in datapoints}  
# var_devs = {p : [] for p in datapoints}
# for dname in stg_empirical:
#     ref_mean, ref_var = stg_empirical[dname]["GAMMA"]
#     for p in datapoints:
#         m = stg_boost[dname][p].mu
#         v = stg_boost[dname][p].var
#         md = 100 - (m / ref_mean)*100
#         mean_devs[p].append(abs(md))
#         vd = 100 - (v / ref_var)*100
#         var_devs[p].append(abs(vd))        

# # Summary.
# with open("{}/stg_moments.txt".format(summary_path), "w") as dest:
#     print("QUALITY OF APPROXIMATION TO THE EXPECTED VALUE AND VARIANCE, RELATIVE TO REFERENCE SOLUTIONS, FOR BOOSTED MC.", file=dest)
#     print("REFERENCE SOLUTIONS COMPUTED VIA MC METHOD WITH 20,000 SAMPLES AND GAMMA WEIGHTS.", file=dest)
#     print("1620 RANDOMLY-GENERATED DAGS BASED ON TOPOLOGIES FROM THE STG.", file=dest)    
            
#     print("\n\n\nDEVIATIONS (%) FROM REFERENCE MOMENTS", file=dest)
#     for p in datapoints:
#         print(p, file=dest)
#         avg = np.mean(mean_devs[p])
#         mx = max(mean_devs[p])
#         print("Mean : avg = {}, max = {}".format(avg, mx), file=dest)
#         avg = np.mean(var_devs[p])
#         mx = max(var_devs[p])
#         print("Variance : avg = {}, max = {}".format(avg, mx), file=dest)
    
#     print("\n\n\nPERCENTAGE OF TIMES BETTER THAN EMPIRICAL (MEAN ESTIMATE, VARIANCE ESTIMATE)", file=dest)  
#     for s in [30, 100]:
#         print("{} samples".format(s), file=dest)
#         msc = sum(1 for s, mc in zip(mean_devs["MC{}-SCULLI".format(s)], mean_devs["MC{}".format(s)]) if s < mc)
#         vsc = sum(1 for s, mc in zip(var_devs["MC{}-SCULLI".format(s)], var_devs["MC{}".format(s)]) if s < mc)
#         print("MC-SCULLI: ({}, {})".format((msc/1620)*100, (vsc/1620)*100), file=dest)
#         mcor = sum(1 for c, mc in zip(mean_devs["MC{}-CorLCA".format(s)], mean_devs["MC{}".format(s)]) if c < mc)
#         vcor = sum(1 for c, mc in zip(var_devs["MC{}-CorLCA".format(s)], var_devs["MC{}".format(s)]) if c < mc)
#         print("MC-CorLCA: ({}, {})".format((mcor/1620)*100, (vcor/1620)*100), file=dest)
#         mmc = sum(1 for x, mc in zip(mean_devs["MC{}-MC".format(s)], mean_devs["MC{}".format(s)]) if x < mc)
#         vmc = sum(1 for x, mc in zip(var_devs["MC{}-MC".format(s)], var_devs["MC{}".format(s)]) if x < mc)
#         print("MC-MC1000: ({}, {})".format((mmc/1620)*100, (vmc/1620)*100), file=dest)         
    

# =============================================================================
# Cholesky.
# =============================================================================

datapoints = ["MC30", "MC30-S", "MC30-C", "MC30-MC", "MC100", "MC100-S", "MC100-C", "MC100-MC"]

# Summary.
# for d1, d2 in it.product(["NORMAL", "GAMMA", "UNIFORM"], ["NORMAL", "GAMMA", "UNIFORM"]):
#     with open("{}/chol_{}_SAMPLING_{}_REFERENCE.txt".format(summary_path, d1, d2), "w") as dest:
#         print("QUALITY OF BOOSTED MC SOLUTIONS.", file=dest)
#         for nt in n_tasks: 
#             print("\n\n\n---------------------------------", file=dest)
#             print("NUMBER OF TASKS: {}".format(nt), file=dest) 
#             print("---------------------------------", file=dest)
            
#             ref_dist = chol_empirical[nt][d2]
#             ref_mean, ref_var = np.mean(ref_dist), np.var(ref_dist) 
#             print("(% rel. error in mean, % rel. error in variance, KS statistic)", file=dest)
#             for h in ["SCULLI", "CorLCA"]:
#                 m, v = chol_existing[nt][h].mu, chol_existing[nt][h].var
#                 me = abs(m - ref_mean)/ref_mean
#                 ve = abs(v - ref_var)/ref_var
#                 ks, _ = kstest(ref_dist, 'norm', args=(m, np.sqrt(v)))
#                 print("{} : ({}, {}, {})".format(h, 100*me, 100*ve, ks), file=dest)
            
#             for p in datapoints:
#                 if p.endswith("MC") or p in ["MC30", "MC100"]:
#                     m, v = np.mean(chol_boost[nt][d1][p]), np.var(chol_boost[nt][d1][p])
#                     ks, _ = ks_2samp(chol_boost[nt][d1][p], ref_dist)
#                 else:
#                     m, v = chol_boost[nt][d1][p].mu, chol_boost[nt][d1][p].var
#                     ks, _ = kstest(ref_dist, 'norm', args=(m, np.sqrt(v)))
#                 me = abs(m - ref_mean)/ref_mean
#                 ve = abs(v - ref_var)/ref_var
#                 print("{} : ({}, {}, {})".format(p, 100*me, 100*ve, ks), file=dest)
                
# Variance plot.
d1, d2 = "NORMAL", "NORMAL"
ks_stats = {d : [] for d in datapoints}
ks_stats["CorLCA"] = []
for nt in n_tasks:
    ref_dist = chol_empirical[nt][d2]
    for h in ["CorLCA"]:
        m, v = chol_existing[nt][h].mu, chol_existing[nt][h].var
        ks, _ = kstest(ref_dist, 'norm', args=(m, np.sqrt(v)))
        ks_stats[h].append(ks)
for s in [30, 100]:
    fig = plt.figure(dpi=400)
    ax1 = fig.add_subplot(111)
    for nt in n_tasks:
        ref_dist = chol_empirical[nt][d2]
        for p in ["MC{}".format(s), "MC{}-S".format(s), "MC{}-C".format(s), "MC{}-MC".format(s)]:
            if p.endswith("MC") or p in ["MC30", "MC100"]:
                m, v = np.mean(chol_boost[nt][d1][p]), np.var(chol_boost[nt][d1][p])
                ks, _ = ks_2samp(chol_boost[nt][d1][p], ref_dist)
            else:
                m, v = chol_boost[nt][d1][p].mu, chol_boost[nt][d1][p].var
                ks, _ = kstest(ref_dist, 'norm', args=(m, np.sqrt(v)))
            ks_stats[p].append(ks)
        
    ax1.plot(n_tasks, ks_stats[h], color='#E24A33', label="CorLCA")
    ax1.plot(n_tasks, ks_stats["MC{}".format(s)], color='#FBC15E', label="MC")
    ax1.plot(n_tasks, ks_stats["MC{}-S".format(s)], color='#8EBA42', label="MC-S")
    ax1.plot(n_tasks, ks_stats["MC{}-C".format(s)], color='#988ED5', label="MC-C")
    ax1.plot(n_tasks, ks_stats["MC{}-MC".format(s)], color='#348ABD', label="MC-MC")
    plt.xscale('log')
    ax1.set_xlabel("DAG SIZE", labelpad=5)
    ax1.set_ylabel("KOLMOGOROV-SMIRNOV STATISTIC", labelpad=5)
    if s == 30:
        ax1.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white') 
    plt.savefig('{}/chol_ks_{}SAMPLES'.format(plot_path, s), bbox_inches='tight') 
    plt.close(fig) 
        
# =============================================================================
# TODO: histogram of MC distributions...
# =============================================================================
        
        
        