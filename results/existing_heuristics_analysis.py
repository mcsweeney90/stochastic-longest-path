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

# # Print full summary.
# with open("{}/all.txt".format(summary_path), "w") as dest:
#     print("BOUNDS AND APPROXIMATIONS TO THE LONGEST PATH DISTRIBUTION.", file=dest) 
#     print("GRAPHS REPRESENT STOCHASTIC SCHEDULES FOR CHOLESKY TASK DAGS.", file=dest)
#     print("SCHEDULES COMPUTED USING THE (STATIC) HEURISTIC HEFT FOR AN ACCELERATED TARGET PLATFORM WITH 7 CPU RESOURCES AND 1 GPU.", file=dest)
#     for nt in n_tasks:   
#         print("\n\n\n---------------------------------", file=dest)
#         print("NUMBER OF TASKS: {}".format(nt), file=dest) 
#         print("---------------------------------", file=dest)
#         print("\nBOUNDS", file=dest) 
#         print("PERT-CPM bound on mean: {}".format(existing[nt]["PERT"]), file=dest)
#         print("Kamburowski bounds on mean: ({}, {})".format(existing[nt]["KML"], existing[nt]["KMU"]), file=dest)
#         print("Kamburowski bounds on variance: ({}, {})".format(existing[nt]["KVL"], existing[nt]["KVU"]), file=dest)
#         print("\nAPPROXIMATIONS", file=dest) 
#         print("Sculli forward: {}".format(existing[nt]["SCULLI"]), file=dest)
#         print("Sculli backward: {}".format(existing[nt]["SCULLI-R"]), file=dest)
#         print("CorLCA forward: {}".format(existing[nt]["CorLCA"]), file=dest)
#         print("CorLCA backward: {}".format(existing[nt]["CorLCA-R"]), file=dest)
#         print("\nMONTE CARLO - NORMAL COSTS", file=dest)
#         m = 10
#         for mcn in existing[nt]["MCN"]:
#             print("{} SAMPLES: {}".format(m, mcn), file=dest) 
#             m *= 10
#         print("\nMONTE CARLO - GAMMA COSTS", file=dest)
#         m = 10
#         for mcg in existing[nt]["MCG"]:
#             print("{} SAMPLES: {}".format(m, mcg), file=dest) 
#             m *= 10

# #Expected value bounds.
# with open("{}/mean_bounds.txt".format(summary_path), "w") as dest:
#     print("TIGHTNESS OF BOUNDS ON THE EXPECTED VALUE RELATIVE TO REFERENCE SOLUTION.", file=dest)
#     for nt in n_tasks: 
#         print("\n\n\n---------------------------------", file=dest)
#         print("NUMBER OF TASKS: {}".format(nt), file=dest) 
#         print("---------------------------------", file=dest)
#         print("\nNORMAL COSTS", file=dest)
#         ref = existing[nt]["MCN"][-1].mu
#         print("Reference solution: {}".format(ref), file=dest)
#         print("PERT-CPM (%): {}".format((existing[nt]["PERT"] /ref)*100), file=dest)
#         print("Kamburowski lower (%): {}".format((existing[nt]["KML"] /ref)*100), file=dest)
#         print("Kamburowski upper (%): {}".format((existing[nt]["KMU"] /ref)*100), file=dest)
        
#         print("\nGAMMA COSTS", file=dest)
#         ref = existing[nt]["MCG"][-1].mu
#         print("Reference solution: {}".format(ref), file=dest)
#         print("PERT-CPM (%): {}".format((existing[nt]["PERT"] /ref)*100), file=dest)
#         print("Kamburowski lower (%): {}".format((existing[nt]["KML"] /ref)*100), file=dest)
#         print("Kamburowski upper (%): {}".format((existing[nt]["KMU"] /ref)*100), file=dest)

# Sculli and CorLCA.
with open("{}/sculli_corlca.txt".format(summary_path), "w") as dest:
    print("QUALITY OF APPROXIMATIONS TO MEAN AND VARIANCE FOR SCULLI AND CORLCA.", file=dest)
    for nt in n_tasks: 
        print("\n\n\n---------------------------------", file=dest)
        print("NUMBER OF TASKS: {}".format(nt), file=dest) 
        print("---------------------------------", file=dest)
        print("\nNORMAL COSTS", file=dest)
        ref_mu = np.mean(empirical[nt]["NORMAL"])
        ref_var = np.var(empirical[nt]["NORMAL"])
        
        print("\nReference mean: {}".format(ref_mu), file=dest)
        print("Sculli mean: {}".format(existing[nt]["SCULLI"].mu), file=dest)
        sd = 100 - (existing[nt]["SCULLI"].mu / ref_mu) * 100
        print("Difference (%): {}".format(sd), file=dest)
        
        print("CorLCA mean: {}".format(existing[nt]["CorLCA"].mu), file=dest)
        cd = 100 - (existing[nt]["CorLCA"].mu / ref_mu) * 100
        print("Difference (%): {}".format(cd), file=dest)
        
        print("\nReference variance: {}".format(ref_var), file=dest)
        print("Sculli variance: {}".format(existing[nt]["SCULLI"].var), file=dest)
        sd = 100 - (existing[nt]["SCULLI"].var / ref_var) * 100
        print("Difference (%): {}".format(sd), file=dest)
        
        print("CorLCA variance: {}".format(existing[nt]["CorLCA"].var), file=dest)
        cd = 100 - (existing[nt]["CorLCA"].var / ref_var) * 100
        print("Difference (%): {}".format(cd), file=dest)
        
        print("\nGAMMA COSTS", file=dest)
        ref_mu = np.mean(empirical[nt]["GAMMA"])
        ref_var = np.var(empirical[nt]["GAMMA"])
        
        print("\nReference mean: {}".format(ref_mu), file=dest)
        print("Sculli mean: {}".format(existing[nt]["SCULLI"].mu), file=dest)
        sd = 100 - (existing[nt]["SCULLI"].mu / ref_mu) * 100
        print("Difference (%): {}".format(sd), file=dest)
        
        print("CorLCA mean: {}".format(existing[nt]["CorLCA"].mu), file=dest)
        cd = 100 - (existing[nt]["CorLCA"].mu / ref_mu) * 100
        print("Difference (%): {}".format(cd), file=dest)
        
        print("\nReference variance: {}".format(ref_var), file=dest)
        print("Sculli variance: {}".format(existing[nt]["SCULLI"].var), file=dest)
        sd = 100 - (existing[nt]["SCULLI"].var / ref_var) * 100
        print("Difference (%): {}".format(sd), file=dest)
        
        print("CorLCA variance: {}".format(existing[nt]["CorLCA"].var), file=dest)
        cd = 100 - (existing[nt]["CorLCA"].var / ref_var) * 100
        print("Difference (%): {}".format(cd), file=dest)
        

# =============================================================================
# Plots.
# =============================================================================

            
# # Variance bounds.
# fig = plt.figure(dpi=400)
# ax1 = fig.add_subplot(111)
# ax1.plot(n_tasks, list(existing[nt]["MCN"][-1].var for nt in n_tasks), color='#E24A33', label="ACTUAL")
# ax1.fill_between(n_tasks, list(existing[nt]["KVL"] for nt in n_tasks), list(existing[nt]["KVU"] for nt in n_tasks), color='#348ABD', alpha=0.5)
# plt.yscale('log')
# ax1.set_xlabel("DAG SIZE", labelpad=5)
# ax1.set_ylabel("VARIANCE", labelpad=5)
# ax1.legend(handlelength=3, handletextpad=0.4, ncol=2, loc='upper left', fancybox=True, facecolor='white') 
# plt.savefig('{}/variance_bounds'.format(plot_path), bbox_inches='tight') 
# plt.close(fig) 
            
