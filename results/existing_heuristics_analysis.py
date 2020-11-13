#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis for before_runtime.py results.
"""

import dill, pathlib, sys
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from collections import defaultdict
sys.path.append('../') # Needed for src apparently...

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
summary_path = "summaries/existing_heuristics"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/existing_heuristics"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

# Cholesky.
nb = 128
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]
with open('data/chol_existing.dill', 'rb') as file:
    chol_existing = dill.load(file) 
with open('data/chol_empirical.dill', 'rb') as file:
    chol_empirical = dill.load(file) 
    
# STG.
with open('data/stg_existing.dill', 'rb') as file:
    stg_existing = dill.load(file) 
with open('data/stg_empirical.dill', 'rb') as file:
    stg_empirical = dill.load(file) 


# =============================================================================
# Full summary for Cholesky DAGs.
# =============================================================================

# with open("{}/chol_full.txt".format(summary_path), "w") as dest:
#     print("BOUNDS AND APPROXIMATIONS TO THE LONGEST PATH DISTRIBUTION.", file=dest) 
#     print("GRAPHS REPRESENT STOCHASTIC SCHEDULES FOR CHOLESKY TASK DAGS.", file=dest)
#     print("SCHEDULES COMPUTED USING THE (STATIC) HEURISTIC HEFT FOR AN ACCELERATED TARGET PLATFORM WITH 7 CPU RESOURCES AND 1 GPU.", file=dest)
#     for nt in n_tasks:   
#         print("\n\n\n---------------------------------", file=dest)
#         print("NUMBER OF TASKS: {}".format(nt), file=dest) 
#         print("---------------------------------", file=dest)
#         print("\nREFERENCE SOLUTIONS", file=dest)
#         print("NORMAL WEIGHTS: RV({}, {})".format(np.mean(chol_empirical[nt]["NORMAL"]), np.var(chol_empirical[nt]["NORMAL"])), file=dest)
#         print("GAMMA WEIGHTS: RV({}, {})".format(np.mean(chol_empirical[nt]["GAMMA"]), np.var(chol_empirical[nt]["GAMMA"])), file=dest)
#         print("\nBOUNDS", file=dest) 
#         print("PERT-CPM bound on mean: {}".format(chol_existing[nt]["PERT"]), file=dest)
#         print("Kamburowski bounds on mean: ({}, {})".format(chol_existing[nt]["KML"], chol_existing[nt]["KMU"]), file=dest)
#         print("Kamburowski bounds on variance: ({}, {})".format(chol_existing[nt]["KVL"], chol_existing[nt]["KVU"]), file=dest)
#         print("\nAPPROXIMATIONS", file=dest) 
#         print("Sculli forward: {}".format(chol_existing[nt]["SCULLI"]), file=dest)
#         print("Sculli backward: {}".format(chol_existing[nt]["SCULLI-R"]), file=dest)
#         print("CorLCA forward: {}".format(chol_existing[nt]["CorLCA"]), file=dest)
#         print("CorLCA backward: {}".format(chol_existing[nt]["CorLCA-R"]), file=dest)
#         for dist in ["NORMAL", "GAMMA"]:
#             print("\nMONTE CARLO - {} COSTS".format(dist), file=dest)
#             for s in [10, 100, 1000, 10000]: 
#                 mu = np.mean(chol_empirical[nt]["NORMAL"][:s])
#                 var = np.var(chol_empirical[nt]["NORMAL"][:s])
#                 print("{} SAMPLES: mu = {}, var = {}".format(s, mu, var), file=dest) 

# =============================================================================
# Cholesky expected value summary.
# =============================================================================

# with open("{}/chol_mean.txt".format(summary_path), "w") as dest:
#     print("TIGHTNESS OF BOUNDS AND APPROXIMATIONS TO THE EXPECTED VALUE, RELATIVE TO REFERENCE SOLUTION.", file=dest)
#     for nt in n_tasks: 
#         print("\n\n\n---------------------------------", file=dest)
#         print("NUMBER OF TASKS: {}".format(nt), file=dest) 
#         print("---------------------------------", file=dest)
#         for dist in ["NORMAL", "GAMMA"]:
#             print("\n{} WEIGHTS".format(dist), file=dest)
#             ref = np.mean(chol_empirical[nt][dist])
#             print("Reference solution: {}".format(ref), file=dest)
#             print("PERT-CPM (%): {}".format((chol_existing[nt]["PERT"] /ref)*100), file=dest)
#             print("Kamburowski lower (%): {}".format((chol_existing[nt]["KML"] /ref)*100), file=dest)
#             print("Kamburowski upper (%): {}".format((chol_existing[nt]["KMU"] /ref)*100), file=dest)
#             print("Sculli (%): {}".format((chol_existing[nt]["SCULLI"].mu /ref)*100), file=dest)
#             print("CorLCA (%): {}".format((chol_existing[nt]["CorLCA"].mu /ref)*100), file=dest)
#             print("MC30 (%): {}".format((chol_existing[nt]["MC30"].mu /ref)*100), file=dest)
#             m = np.mean(chol_empirical[nt][dist][:10])
#             print("MC10 (%): {}".format((m /ref)*100), file=dest)        
        
# =============================================================================
# Cholesky variance summary.
# =============================================================================
        
# with open("{}/chol_variance.txt".format(summary_path), "w") as dest:
#     print("TIGHTNESS OF BOUNDS AND APPROXIMATIONS TO THE VARIANCE, RELATIVE TO REFERENCE SOLUTION.", file=dest)
#     for nt in n_tasks: 
#         print("\n\n\n---------------------------------", file=dest)
#         print("NUMBER OF TASKS: {}".format(nt), file=dest) 
#         print("---------------------------------", file=dest)
#         for dist in ["NORMAL", "GAMMA"]:
#             print("\n{} WEIGHTS".format(dist), file=dest)
#             ref = np.var(chol_empirical[nt][dist])
#             print("Reference solution: {}".format(ref), file=dest)
#             print("Kamburowski lower (%): {}".format((chol_existing[nt]["KVL"] /ref)*100), file=dest)
#             print("Kamburowski upper (%): {}".format((chol_existing[nt]["KVU"] /ref)*100), file=dest)
#             print("Sculli (%): {}".format((chol_existing[nt]["SCULLI"].var /ref)*100), file=dest)
#             print("CorLCA (%): {}".format((chol_existing[nt]["CorLCA"].var /ref)*100), file=dest)
#             print("MC30 (%): {}".format((chol_existing[nt]["MC30"].var /ref)*100), file=dest)

# =============================================================================
# Variance bounds/approximations for Cholesky (plot).
# =============================================================================

# empirical_vars = list(np.var(chol_empirical[nt]["GAMMA"]) for nt in n_tasks)
# fig = plt.figure(dpi=400)
# ax1 = fig.add_subplot(111)
# ax1.plot(n_tasks, empirical_vars, color='#E24A33', label="ACTUAL")
# ax1.plot(n_tasks, list(chol_existing[nt]["SCULLI"].var for nt in n_tasks), color='#8EBA42', label="SCULLI")
# ax1.plot(n_tasks, list(chol_existing[nt]["CorLCA"].var for nt in n_tasks), color='#988ED5', label="CorLCA")
# ax1.plot(n_tasks, list(chol_existing[nt]["MC30"].var for nt in n_tasks), color='#FBC15E', label="MC30")
# ax1.fill_between(n_tasks, list(chol_existing[nt]["KVL"] for nt in n_tasks), list(chol_existing[nt]["KVU"] for nt in n_tasks), color='#348ABD', alpha=0.3)
# plt.yscale('log')
# ax1.set_xlabel("DAG SIZE", labelpad=5)
# ax1.set_ylabel("VARIANCE", labelpad=5)
# ax1.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='best', fancybox=True, facecolor='white') 
# plt.savefig('{}/chol_variance'.format(plot_path), bbox_inches='tight') 
# plt.close(fig) 

# =============================================================================
# Timing plots for Cholesky. 
# =============================================================================

# with open('data/chol_existing_timings.dill', 'rb') as file:
#     chol_timings = dill.load(file) 

# colors = {"KAMBUROWSKI" : '#348ABD', "SCULLI" : '#8EBA42', "CorLCA" : '#988ED5', "MC30" : '#FBC15E'}
# fig = plt.figure(dpi=400)
# ax1 = fig.add_subplot(111)
# for p in ["KAMBUROWSKI", "SCULLI", "CorLCA", "MC30"]:    
#     ax1.plot(n_tasks, chol_timings[p], color=colors[p], label=p)
# # plt.yscale('log')
# ax1.set_xlabel("DAG SIZE", labelpad=5)
# ax1.set_ylabel("TIME (SECONDS)", labelpad=5)
# ax1.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white') 
# plt.savefig('{}/chol_existing_timings'.format(plot_path), bbox_inches='tight') 
# plt.close(fig)

# =============================================================================
# Mean estimates for STG set.
# =============================================================================

# devs = {p : [] for p in ["PERT", "KML", "KMU", "SCULLI", "CorLCA", "MC30"]}      
# abs_devs = {p : [] for p in ["PERT", "KML", "KMU", "SCULLI", "CorLCA", "MC30"]}
# for dname in stg_empirical:
#     ref = stg_empirical[dname]["GAMMA"][0]
#     for p in ["PERT", "KML", "KMU", "SCULLI", "CorLCA", "MC30"]:
#         try:
#             m = stg_existing[dname][p].mu
#         except AttributeError:
#             m = stg_existing[dname][p]
#         d = 100 - (m / ref)*100
#         devs[p].append(d)
#         abs_devs[p].append(abs(d))

# # Summary.
# with open("{}/stg_mean.txt".format(summary_path), "w") as dest:
#     print("TIGHTNESS OF BOUNDS AND APPROXIMATIONS TO THE EXPECTED VALUE, RELATIVE TO REFERENCE SOLUTION.", file=dest)
#     print("REFERENCE SOLUTIONS COMPUTED VIS MC METHOD WITH 20,000 SAMPLES AND GAMMA WEIGHTS.", file=dest)
#     print("1620 RANDOMLY-GENERATED DAGS BASED ON TOPOLOGIES FROM THE STG.", file=dest)    
            
#     print("\n\n\nDEVIATIONS (%) FROM TRUE MEAN", file=dest)
#     for p in ["PERT", "KML", "KMU", "SCULLI", "CorLCA", "MC30"]:
#         avg = np.mean(abs_devs[p])
#         mx = max(abs_devs[p])
#         print("{} : avg = {}, max = {}".format(p, avg, mx), file=dest)
    
#     print("\n\n\nPERCENTAGE OF TIMES BETTER", file=dest)         
#     cvs = sum(1 for c, s in zip(abs_devs["CorLCA"], abs_devs["SCULLI"]) if c < s)
#     print("CorLCA vs SCULLI : {}".format((cvs/1620)*100), file=dest)
#     cvm = sum(1 for c, mc in zip(abs_devs["CorLCA"], abs_devs["MC30"]) if c < mc)
#     print("CorLCA vs MC30 : {} ".format((cvm/1620)*100), file=dest)
#     mvs = sum(1 for mc, s in zip(abs_devs["MC30"], abs_devs["SCULLI"]) if mc < s)
#     print("MC30 vs Sculli : {} ".format((mvs/1620)*100), file=dest)
    
#     lower_viols = sum(1 for d in devs["KML"] if d < 0.0)
#     max_lower_viol = abs(min(devs["KML"]))
#     upper_viols = sum(1 for d in devs["KMU"] if d > 0.0)
#     max_upper_viol = max(devs["KMU"])
#     print("\n\n\nVIOLATIONS OF KAMBUROWSKI'S BOUNDS (%, worst)", file=dest)
#     print("UPPER: ({}, {})".format((upper_viols/1620)*100, max_upper_viol), file=dest)
#     print("LOWER: ({}, {})".format((lower_viols/1620)*100, max_lower_viol), file=dest)
    
# # Histogram.
# sols = ["PERT", "KML", "SCULLI", "CorLCA", "MC30"]
# avgs = [np.mean(abs_devs[p]) for p in sols]    
# x = np.arange(len(sols))
# colors = ['#E24A33', '#348ABD', '#8EBA42', '#988ED5', '#FBC15E']
# fig = plt.figure(dpi=400)
# ax1 = fig.add_subplot(111)
# ax1.bar(sols, avgs, color=colors, edgecolor='white')             
# ax1.set_xticks(x)
# ax1.set_xticklabels(["CPM", "K. LOWER", "SCULLI", "CorLCA", "MC30"]) 
# ax1.set_ylabel("AVERAGE DEVIATION (%)", labelpad=5)
# # ax1.set_title("MEAN", weight='bold')
# plt.savefig('{}/stg_existing_mean'.format(plot_path), bbox_inches='tight') 
# plt.close(fig) 
    
# =============================================================================
# Variance estimates for STG set.
# =============================================================================

devs = {p : [] for p in ["KVL", "KVU", "SCULLI", "CorLCA", "MC30"]}      
abs_devs = {p : [] for p in ["KVL", "KVU", "SCULLI", "CorLCA", "MC30"]} 
for dname in stg_empirical:
    ref = stg_empirical[dname]["GAMMA"][1]
    for p in ["KVL", "KVU", "SCULLI", "CorLCA", "MC30"]:
        try:
            v = stg_existing[dname][p].var
        except AttributeError:
            v = stg_existing[dname][p]
        d = 100 - (v / ref)*100
        devs[p].append(d)
        abs_devs[p].append(abs(d))

# with open("{}/stg_variance.txt".format(summary_path), "w") as dest:
#     print("TIGHTNESS OF BOUNDS AND APPROXIMATIONS TO THE VARIANCE, RELATIVE TO REFERENCE SOLUTION.", file=dest)
#     print("REFERENCE SOLUTIONS COMPUTED VIS MC METHOD WITH 20,000 SAMPLES AND GAMMA WEIGHTS.", file=dest)
#     print("1620 RANDOMLY-GENERATED DAGS BASED ON TOPOLOGIES FROM THE STG.", file=dest)
                
#     print("\n\n\nDEVIATIONS (%) FROM TRUE VARIANCE", file=dest)
#     for p in ["KVL", "KVU", "SCULLI", "CorLCA", "MC30"]:
#         avg = np.mean(abs_devs[p])
#         mx = max(abs_devs[p])
#         print("{} : avg = {}, max = {}".format(p, avg, mx), file=dest)   
        
#     print("\n\n\nPERCENTAGE OF TIMES BETTER", file=dest)         
#     cvs = sum(1 for c, s in zip(abs_devs["CorLCA"], abs_devs["SCULLI"]) if c < s)
#     print("CorLCA vs SCULLI : {}".format((cvs/1620)*100), file=dest)
#     cvm = sum(1 for c, mc in zip(abs_devs["CorLCA"], abs_devs["MC30"]) if c < mc)
#     print("CorLCA vs MC30 : {} ".format((cvm/1620)*100), file=dest)
#     mvs = sum(1 for mc, s in zip(abs_devs["MC30"], abs_devs["SCULLI"]) if mc < s)
#     print("MC30 vs Sculli : {} ".format((mvs/1620)*100), file=dest)
    
#     lower_viols = sum(1 for d in devs["KVL"] if d < 0.0)
#     max_lower_viol = abs(min(devs["KVL"]))
#     upper_viols = sum(1 for d in devs["KVU"] if d > 0.0)
#     max_upper_viol = max(devs["KVU"])
#     print("\n\n\nVIOLATIONS OF KAMBUROWSKI'S BOUNDS (%, worst)", file=dest)
#     print("UPPER: ({}, {})".format((upper_viols/1620)*100, max_upper_viol), file=dest)
#     print("LOWER: ({}, {})".format((lower_viols/1620)*100, max_lower_viol), file=dest)      

# Histogram.
sols = ["KVL", "KVU", "SCULLI", "CorLCA", "MC30"]
avgs = [np.mean(abs_devs[p]) for p in sols]    
x = np.arange(len(sols))
colors = ['#E24A33', '#348ABD', '#8EBA42', '#988ED5', '#FBC15E']
fig = plt.figure(dpi=400)
ax1 = fig.add_subplot(111)
ax1.bar(sols, avgs, color=colors, edgecolor='white')             
ax1.set_xticks(x)
ax1.set_xticklabels(["K. LOWER", "K. UPPER", "SCULLI", "CorLCA", "MC30"]) 
ax1.set_ylabel("AVERAGE DEVIATION (%)", labelpad=5)
# ax1.set_title("VARIANCE", weight='bold')
plt.savefig('{}/stg_existing_var'.format(plot_path), bbox_inches='tight') 
plt.close(fig) 
            
