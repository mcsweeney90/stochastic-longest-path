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

####################################################################################################

# Destinations to save summaries and generated plots.
summary_path = "summaries/existing_heuristics"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/existing_heuristics"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

nb = 128
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]

# Load info.
with open('existing_heuristics.dill', 'rb') as file:
    existing = dill.load(file) 
with open('empirical_dists.dill', 'rb') as file:
    empirical = dill.load(file) 

# =============================================================================
# Summaries.
# =============================================================================

# Print full summary.
with open("{}/all.txt".format(summary_path), "w") as dest:
    print("BOUNDS AND APPROXIMATIONS TO THE LONGEST PATH DISTRIBUTION.", file=dest) 
    print("GRAPHS REPRESENT STOCHASTIC SCHEDULES FOR CHOLESKY TASK DAGS.", file=dest)
    print("SCHEDULES COMPUTED USING THE (STATIC) HEURISTIC HEFT FOR AN ACCELERATED TARGET PLATFORM WITH 7 CPU RESOURCES AND 1 GPU.", file=dest)
    for nt in n_tasks:   
        print("\n\n\n---------------------------------", file=dest)
        print("NUMBER OF TASKS: {}".format(nt), file=dest) 
        print("---------------------------------", file=dest)
        print("\nREFERENCE SOLUTIONS", file=dest)
        print("NORMAL WEIGHTS: RV({}, {})".format(np.mean(empirical[nt]["NORMAL"]), np.var(empirical[nt]["NORMAL"])), file=dest)
        print("GAMMA WEIGHTS: RV({}, {})".format(np.mean(empirical[nt]["GAMMA"]), np.var(empirical[nt]["GAMMA"])), file=dest)
        print("\nBOUNDS", file=dest) 
        print("PERT-CPM bound on mean: {}".format(existing[nt]["PERT"]), file=dest)
        print("Kamburowski bounds on mean: ({}, {})".format(existing[nt]["KML"], existing[nt]["KMU"]), file=dest)
        print("Kamburowski bounds on variance: ({}, {})".format(existing[nt]["KVL"], existing[nt]["KVU"]), file=dest)
        print("\nAPPROXIMATIONS", file=dest) 
        print("Sculli forward: {}".format(existing[nt]["SCULLI"]), file=dest)
        print("Sculli backward: {}".format(existing[nt]["SCULLI-R"]), file=dest)
        print("CorLCA forward: {}".format(existing[nt]["CorLCA"]), file=dest)
        print("CorLCA backward: {}".format(existing[nt]["CorLCA-R"]), file=dest)
        print("\nMONTE CARLO - NORMAL COSTS", file=dest)
        m = 10
        for mcn in existing[nt]["MCN"][:-1]:
            print("{} SAMPLES: {}".format(m, mcn), file=dest) 
            m *= 10
        print("\nMONTE CARLO - GAMMA COSTS", file=dest)
        m = 10
        for mcg in existing[nt]["MCG"][:-1]:
            print("{} SAMPLES: {}".format(m, mcg), file=dest) 
            m *= 10

# #Expected value.
# with open("{}/mean.txt".format(summary_path), "w") as dest:
#     print("TIGHTNESS OF BOUNDS AND APPROXIMATIONS TO THE EXPECTED VALUE, RELATIVE TO REFERENCE SOLUTION.", file=dest)
#     for nt in n_tasks: 
#         print("\n\n\n---------------------------------", file=dest)
#         print("NUMBER OF TASKS: {}".format(nt), file=dest) 
#         print("---------------------------------", file=dest)
#         print("\nNORMAL COSTS", file=dest)
#         ref = np.mean(empirical[nt]["NORMAL"])
#         print("Reference solution: {}".format(ref), file=dest)
#         print("PERT-CPM (%): {}".format((existing[nt]["PERT"] /ref)*100), file=dest)
#         print("Kamburowski lower (%): {}".format((existing[nt]["KML"] /ref)*100), file=dest)
#         print("Kamburowski upper (%): {}".format((existing[nt]["KMU"] /ref)*100), file=dest)
#         print("Sculli (%): {}".format((existing[nt]["SCULLI"].mu /ref)*100), file=dest)
#         print("CorLCA (%): {}".format((existing[nt]["CorLCA"].mu /ref)*100), file=dest)
        
#         print("\nGAMMA COSTS", file=dest)
#         ref = np.mean(empirical[nt]["GAMMA"])
#         print("Reference solution: {}".format(ref), file=dest)
#         print("PERT-CPM (%): {}".format((existing[nt]["PERT"] /ref)*100), file=dest)
#         print("Kamburowski lower (%): {}".format((existing[nt]["KML"] /ref)*100), file=dest)
#         print("Kamburowski upper (%): {}".format((existing[nt]["KMU"] /ref)*100), file=dest)
#         print("Sculli (%): {}".format((existing[nt]["SCULLI"].mu /ref)*100), file=dest)
#         print("CorLCA (%): {}".format((existing[nt]["CorLCA"].mu /ref)*100), file=dest)
        
# #Variance.
# with open("{}/variance.txt".format(summary_path), "w") as dest:
#     print("TIGHTNESS OF BOUNDS AND APPROXIMATIONS TO THE VARIANCE, RELATIVE TO REFERENCE SOLUTION.", file=dest)
#     for nt in n_tasks: 
#         print("\n\n\n---------------------------------", file=dest)
#         print("NUMBER OF TASKS: {}".format(nt), file=dest) 
#         print("---------------------------------", file=dest)
#         print("\nNORMAL COSTS", file=dest)
#         ref = np.var(empirical[nt]["NORMAL"])
#         print("Reference solution: {}".format(ref), file=dest)
#         print("Kamburowski lower (%): {}".format((existing[nt]["KVL"] /ref)*100), file=dest)
#         print("Kamburowski upper (%): {}".format((existing[nt]["KVU"] /ref)*100), file=dest)
#         print("Sculli (%): {}".format((existing[nt]["SCULLI"].var /ref)*100), file=dest)
#         print("CorLCA (%): {}".format((existing[nt]["CorLCA"].var /ref)*100), file=dest)
        
#         print("\nGAMMA COSTS", file=dest)
#         ref = np.var(empirical[nt]["GAMMA"])
#         print("Reference solution: {}".format(ref), file=dest)
#         print("Kamburowski lower (%): {}".format((existing[nt]["KVL"] /ref)*100), file=dest)
#         print("Kamburowski upper (%): {}".format((existing[nt]["KVU"] /ref)*100), file=dest)
#         print("Sculli (%): {}".format((existing[nt]["SCULLI"].var /ref)*100), file=dest)
#         print("CorLCA (%): {}".format((existing[nt]["CorLCA"].var /ref)*100), file=dest)
        

# =============================================================================
# Plots.
# =============================================================================

# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            
# # Variance.
# empirical_vars = list(np.var(empirical[nt]["NORMAL"]) for nt in n_tasks)
# fig = plt.figure(dpi=400)
# ax1 = fig.add_subplot(111)
# ax1.plot(n_tasks, empirical_vars, color='#E24A33', label="ACTUAL")
# ax1.plot(n_tasks, list(existing[nt]["SCULLI"].var for nt in n_tasks), color='#8EBA42', label="SCULLI")
# ax1.plot(n_tasks, list(existing[nt]["CorLCA"].var for nt in n_tasks), color='#988ED5', label="CorLCA")
# ax1.fill_between(n_tasks, list(existing[nt]["KVL"] for nt in n_tasks), list(existing[nt]["KVU"] for nt in n_tasks), color='#348ABD', alpha=0.3)
# plt.yscale('log')
# ax1.set_xlabel("DAG SIZE", labelpad=5)
# ax1.set_ylabel("VARIANCE", labelpad=5)
# ax1.legend(handlelength=3, handletextpad=0.4, ncol=3, loc='upper left', fancybox=True, facecolor='white') 
# plt.savefig('{}/variance_bounds'.format(plot_path), bbox_inches='tight') 
# plt.close(fig) 

# # Timings.
# timings = {"KAMBUROWSKI" : [0.016005605459213257, 0.17476089671254158, 0.6470920592546463, 1.5658568777143955, 3.069533459842205, 5.308403853327036, 8.511513352394104, 12.796787660568953],
#            "SCULLI" : [0.007865753024816513, 0.08773763850331306, 0.329572681337595, 0.7906846031546593, 1.5681733973324299, 2.6994655318558216, 4.334559187293053, 6.578559648245573],
#            "CorLCA" : [0.008354179561138153, 0.09211373329162598, 0.3452003374695778, 0.8503920026123524, 1.7388480640947819, 3.15724578499794, 5.352320522069931, 8.862768094986677]
#            }
# colors = {"KAMBUROWSKI" : '#E24A33', "SCULLI" : '#348ABD', "CorLCA" : '#988ED5'}
# fig = plt.figure(dpi=400)
# ax1 = fig.add_subplot(111)
# for p in ["KAMBUROWSKI", "SCULLI", "CorLCA"]:    
#     ax1.plot(n_tasks, timings[p], color=colors[p], label=p)
# # plt.yscale('log')
# ax1.set_xlabel("DAG SIZE", labelpad=5)
# ax1.set_ylabel("TIME (SECONDS)", labelpad=5)
# ax1.legend(handlelength=3, handletextpad=0.4, ncol=1, loc='best', fancybox=True, facecolor='white') 
# plt.savefig('{}/existing_timings'.format(plot_path), bbox_inches='tight') 
# plt.close(fig)

# for st, ct in zip(timings["SCULLI"], timings["CorLCA"]):
#     print(ct/st * 100)
                  
            
