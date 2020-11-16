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
summary_path = "summaries/path_heuristics"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/path_heuristics"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

# Cholesky.
nb = 128
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]
with open('data/chol_path.dill', 'rb') as file:
    chol_path = dill.load(file) 
with open('data/chol_existing.dill', 'rb') as file:
    chol_existing = dill.load(file) 
with open('data/chol_empirical.dill', 'rb') as file:
    chol_empirical = dill.load(file) 
    
# STG.
with open('data/stg_path.dill', 'rb') as file:
    stg_path = dill.load(file) 
with open('data/stg_existing.dill', 'rb') as file:
    stg_existing = dill.load(file) 
with open('data/stg_empirical.dill', 'rb') as file:
    stg_empirical = dill.load(file) 

# =============================================================================
# Mean estimates for STG set.
# =============================================================================

devs = {p : [] for p in ["MC30", "MC30-SCULLI", "MC30-CorLCA"]}      
abs_devs = {p : [] for p in ["MC30", "MC30-SCULLI", "MC30-CorLCA"]}
for dname in stg_empirical:
    ref = stg_empirical[dname]["GAMMA"][0]
    for p in ["MC30", "MC30-SCULLI", "MC30-CorLCA"]:
        m = stg_path[dname][p].mu
        d = 100 - (m / ref)*100
        devs[p].append(d)
        abs_devs[p].append(abs(d))

# Summary.
with open("{}/stg_mean.txt".format(summary_path), "w") as dest:
    print("QUALITY OF APPROXIMATION TO THE EXPECTED VALUE, RELATIVE TO REFERENCE SOLUTION, FOR PATH-CENTRIC VERSIONS OF SCULLI AND CORLCA.", file=dest)
    print("REFERENCE SOLUTIONS COMPUTED VIA MC METHOD WITH 20,000 SAMPLES AND GAMMA WEIGHTS.", file=dest)
    print("1620 RANDOMLY-GENERATED DAGS BASED ON TOPOLOGIES FROM THE STG.", file=dest)    
            
    print("\n\n\nDEVIATIONS (%) FROM TRUE MEAN", file=dest)
    for p in ["MC30", "MC30-SCULLI", "MC30-CorLCA"]:
        avg = np.mean(abs_devs[p])
        mx = max(abs_devs[p])
        print("{} : avg = {}, max = {}".format(p, avg, mx), file=dest)
    
    print("\n\n\nPERCENTAGE OF TIMES BETTER", file=dest)         
    scu = sum(1 for s, mc in zip(abs_devs["MC30-SCULLI"], abs_devs["MC30"]) if s < mc)
    print("MC-SCULLI VS MC: {}".format((scu/1620)*100), file=dest)
    cor = sum(1 for c, mc in zip(abs_devs["MC30-CorLCA"], abs_devs["MC30"]) if c < mc)
    print("MC-CorLCA VS MC: {}".format((cor/1620)*100), file=dest)    
    
# =============================================================================
# Variance estimates for STG set.
# =============================================================================

devs = {p : [] for p in ["MC30", "MC30-SCULLI", "MC30-CorLCA"]}      
abs_devs = {p : [] for p in ["MC30", "MC30-SCULLI", "MC30-CorLCA"]}
for dname in stg_empirical:
    ref = stg_empirical[dname]["GAMMA"][1]
    for p in ["MC30", "MC30-SCULLI", "MC30-CorLCA"]:
        v = stg_path[dname][p].var
        d = 100 - (v / ref)*100
        devs[p].append(d)
        abs_devs[p].append(abs(d))

# Summary.
with open("{}/stg_var.txt".format(summary_path), "w") as dest:
    print("QUALITY OF APPROXIMATION TO THE VARIANCE, RELATIVE TO REFERENCE SOLUTION, FOR PATH-CENTRIC VERSIONS OF SCULLI AND CORLCA.", file=dest)
    print("REFERENCE SOLUTIONS COMPUTED VIA MC METHOD WITH 20,000 SAMPLES AND GAMMA WEIGHTS.", file=dest)
    print("1620 RANDOMLY-GENERATED DAGS BASED ON TOPOLOGIES FROM THE STG.", file=dest)    
            
    print("\n\n\nDEVIATIONS (%) FROM TRUE VARIANCE", file=dest)
    for p in ["MC30", "MC30-SCULLI", "MC30-CorLCA"]:
        avg = np.mean(abs_devs[p])
        mx = max(abs_devs[p])
        print("{} : avg = {}, max = {}".format(p, avg, mx), file=dest)
    
    print("\n\n\nPERCENTAGE OF TIMES BETTER", file=dest)         
    scu = sum(1 for s, mc in zip(abs_devs["MC30-SCULLI"], abs_devs["MC30"]) if s < mc)
    print("MC-SCULLI VS MC: {}".format((scu/1620)*100), file=dest)
    cor = sum(1 for c, mc in zip(abs_devs["MC30-CorLCA"], abs_devs["MC30"]) if c < mc)
    print("MC-CorLCA VS MC: {}".format((cor/1620)*100), file=dest)
    

# =============================================================================
# Cholesky expected value summary.
# =============================================================================

with open("{}/chol_mean.txt".format(summary_path), "w") as dest:
    print("QUALITY OF APPROXIMATION TO THE EXPECTED VALUE, RELATIVE TO REFERENCE SOLUTION, FOR PATH-CENTRIC VERSIONS OF SCULLI AND CORLCA.", file=dest)
    for nt in n_tasks: 
        print("\n\n\n---------------------------------", file=dest)
        print("NUMBER OF TASKS: {}".format(nt), file=dest) 
        print("---------------------------------", file=dest)
        
        ref = np.mean(chol_empirical[nt]["GAMMA"])
        print("Reference solution: {}".format(ref), file=dest)
        print("Sculli (%): {}".format((chol_existing[nt]["SCULLI"].mu /ref)*100), file=dest)
        print("CorLCA (%): {}".format((chol_existing[nt]["CorLCA"].mu /ref)*100), file=dest)
        print("MC30 (%): {}".format((chol_path[nt]["MC30"].mu /ref)*100), file=dest)
        print("MC30-SCULLI (%): {}".format((chol_path[nt]["MC30-SCULLI"].mu /ref)*100), file=dest)
        print("MC30-CorLCA (%): {}".format((chol_path[nt]["MC30-CorLCA"].mu /ref)*100), file=dest)

# =============================================================================
# Cholesky variance summary.
# =============================================================================

with open("{}/chol_var.txt".format(summary_path), "w") as dest:
    print("QUALITY OF APPROXIMATION TO THE VARIANCE, RELATIVE TO REFERENCE SOLUTION, FOR PATH-CENTRIC VERSIONS OF SCULLI AND CORLCA.", file=dest)
    for nt in n_tasks: 
        print("\n\n\n---------------------------------", file=dest)
        print("NUMBER OF TASKS: {}".format(nt), file=dest) 
        print("---------------------------------", file=dest)
        
        ref = np.var(chol_empirical[nt]["GAMMA"])
        print("Reference solution: {}".format(ref), file=dest)
        print("Sculli (%): {}".format((chol_existing[nt]["SCULLI"].var /ref)*100), file=dest)
        print("CorLCA (%): {}".format((chol_existing[nt]["CorLCA"].var /ref)*100), file=dest)
        print("MC30 (%): {}".format((chol_path[nt]["MC30"].var /ref)*100), file=dest)
        print("MC30-SCULLI (%): {}".format((chol_path[nt]["MC30-SCULLI"].var /ref)*100), file=dest)
        print("MC30-CorLCA (%): {}".format((chol_path[nt]["MC30-CorLCA"].var /ref)*100), file=dest)
        
        
        