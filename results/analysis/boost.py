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
from collections import defaultdict
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

datapoints = ["MC30", "MC30-SCULLI", "MC30-CorLCA", "MC30-MC", "MC100", "MC100-SCULLI", "MC100-CorLCA", "MC100-MC"]

# Summary.
with open("{}/chol_moments.txt".format(summary_path), "w") as dest:
    print("QUALITY OF APPROXIMATION TO THE EXPECTED VALUE AND VARIANCE, RELATIVE TO REFERENCE SOLUTIONS, FOR BOOSTED MC.", file=dest)
    for nt in n_tasks: 
        print("\n\n\n---------------------------------", file=dest)
        print("NUMBER OF TASKS: {}".format(nt), file=dest) 
        print("---------------------------------", file=dest)
        
        ref_mean, ref_var = np.mean(chol_empirical[nt]["GAMMA"]), np.var(chol_empirical[nt]["GAMMA"])
        print("Reference solution: mu = {}, var = {}".format(ref_mean, ref_var), file=dest)
        print("Sculli (%): ({}, {})".format((chol_existing[nt]["SCULLI"].mu/ref_mean)*100, (chol_existing[nt]["SCULLI"].var/ref_var)*100), file=dest)
        print("CorLCA (%): ({}, {})".format((chol_existing[nt]["CorLCA"].mu/ref_mean)*100, (chol_existing[nt]["CorLCA"].var/ref_var)*100), file=dest)
        for p in datapoints:
            if p.endswith("MC"):
                m = np.mean(chol_boost[nt][p])
                v = np.var(chol_boost[nt][p])
            else:
                m = chol_boost[nt][p].mu
                v = chol_boost[nt][p].var
            print("\n{} (%): ({}, {})".format(p, (m/ref_mean)*100, (v/ref_var)*100), file=dest) 
            if p not in ["MC30", "MC100"]:
                print("Time taken: {}".format(chol_boost_timings[nt][p]), file=dest) 
            
# Variance plot.
empirical_vars = list(np.var(chol_empirical[nt]["GAMMA"]) for nt in n_tasks)
fig = plt.figure(dpi=400)
ax1 = fig.add_subplot(111)
ax1.plot(n_tasks, empirical_vars, color='#E24A33', label="ACTUAL")
ax1.plot(n_tasks, list(chol_boost[nt]["MC30"].var for nt in n_tasks), color='#FBC15E', label="MC30")
ax1.plot(n_tasks, list(chol_boost[nt]["MC30-SCULLI"].var for nt in n_tasks), color='#8EBA42', label="MC30-S")
ax1.plot(n_tasks, list(chol_boost[nt]["MC30-CorLCA"].var for nt in n_tasks), color='#988ED5', label="MC30-C")
ax1.plot(n_tasks, list(np.var(chol_boost[nt]["MC30-MC"]) for nt in n_tasks), color='#348ABD', label="MC30-MC")
# plt.yscale('log')
ax1.set_xlabel("DAG SIZE", labelpad=5)
ax1.set_ylabel("VARIANCE", labelpad=5)
ax1.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white') 
plt.savefig('{}/chol_boost_var'.format(plot_path), bbox_inches='tight') 
plt.close(fig) 
        
# =============================================================================
# TODO: histogram of MC distributions...
# =============================================================================
        
        
        