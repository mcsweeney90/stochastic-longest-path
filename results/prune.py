#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 22:16:29 2020

@author: tom
"""

import dill, pathlib, sys, os
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from scipy.stats import norm
from collections import defaultdict
sys.path.append('../') 
from src import RV, SDAG, path_max

data_dest = "data"

# =============================================================================
# Cholesky.
# =============================================================================

chol_dag_path = '../graphs/cholesky'
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]

info, timings = {}, {}

for nt in n_tasks:
    with open('{}/nb128/{}tasks.dill'.format(chol_dag_path, nt), 'rb') as file:
        G = dill.load(file)
    H = SDAG(G)
    timings[nt], info[nt] = {}, {}
    
    for g in [0, 0.5, 0.7, 0.9]:
        timings[nt][g] = {}
        info[nt][g] = {}
        
        # Get critical subgraph.
        start = timer()
        C = H.get_critical_subgraph(gamma=g, weights="mean")
        elapsed = timer() - start
        timings[nt][g]["CREATE SUBGRAPH"] = elapsed    
        
        # Do the Monte Carlo.
        for dist in ["normal", "gamma", "uniform"]:
            start = timer()
            emp = C.monte_carlo(samples=1000, dist=dist)
            elapsed = timer() - start
            timings[nt][g]["MC1000-{}".format(dist)] = elapsed
            info[nt][g][dist] = emp   
                       
    
# Save the info dict.
with open('{}/chol_prune.dill'.format(data_dest), 'wb') as handle:
    dill.dump(info, handle)  
with open('{}/chol_prune_timings.dill'.format(data_dest), 'wb') as handle:
    dill.dump(timings, handle) 