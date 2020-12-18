#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of data gathered in paths.py. 
"""

import dill, pathlib, sys
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from collections import defaultdict
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
summary_path = "../summaries/paths"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "../plots/paths"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

# Cholesky.
nb = 128
n_tasks = [35, 220, 680, 1540, 2925]#, 4960, 7770, 11480]
with open('../data/chol_paths.dill', 'rb') as file:
    chol_paths = dill.load(file) 
with open('../data/chol_empirical.dill', 'rb') as file:
    chol_empirical = dill.load(file) 
    
# =============================================================================
# Cholesky.
# =============================================================================

# Full summary.
with open("{}/chol.txt".format(summary_path), "w") as dest:
    print("QUALITY OF PATH BASED MC SOLUTIONS RELATIVE TO FULL MC SOLUTIONS.", file=dest)
    for nt in n_tasks: 
        print("\n\n\n---------------------------------", file=dest)
        print("NUMBER OF TASKS: {}".format(nt), file=dest) 
        print("---------------------------------", file=dest)
        
        print("------------------------------------------------------", file=dest)
        
        emp_dist = chol_paths[nt]["DIST"]
        emp_mean, emp_var = np.mean(emp_dist), np.var(emp_dist) 
        
        for dist in ["NORMAL", "GAMMA", "UNIFORM"]:
            ref_dist = chol_empirical[nt][dist]            
            ref_mean, ref_var = np.mean(ref_dist), np.var(ref_dist)         
            print("\n{} REFERENCE".format(dist), file=dest)
                   
            ks, p = ks_2samp(emp_dist, ref_dist)
            print("2-sided KS test: ({}, {})".format(ks, p), file=dest)
            mu_rel = abs(emp_mean - ref_mean)/ref_mean
            print("Relative error in mean (%): {}".format(mu_rel*100), file=dest)
            var_rel = abs(emp_var - ref_var)/ref_var
            print("Relative error in variance (%): {}".format(var_rel*100), file=dest)
            # path_time = sum(chol_paths[nt]["TIME"])
            # full_time = 0 # TODO - where was this recorded?
            # time_reduction = 100 - (prune_time/full_time)*100
            # print("Time savings (%): {}".format(time_reduction), file=dest)
            print("------------------------------------------------------", file=dest)