#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path-based methods.
"""

import dill, sys, os
import networkx as nx
import numpy as np
from timeit import default_timer as timer
from scipy.stats import ks_2samp
sys.path.append('../') 
from src import RV, SDAG, path_max

data_dest = "data"

# =============================================================================
# Cholesky.
# =============================================================================

chol_dag_path = '../graphs/cholesky'
n_tasks = [35, 220, 680, 1540, 2925]#, 4960]#, 7770, 11480]
s = 1000
info = {}

for nt in n_tasks:
    with open('{}/nb128/{}tasks.dill'.format(chol_dag_path, nt), 'rb') as file:
        G = dill.load(file)
    H = SDAG(G)
    info[nt] = {}
    info[nt]["TIME"] = []
    print("\nNumber of tasks: {}".format(nt))
    
    start = timer()
    P = H.dodin_critical_paths(epsilon=0.05, K=100, correlations=True)    
    elapsed = timer() - start
    print("Time taken: {}".format(elapsed))
    info[nt]["TIME"].append(elapsed)
    
    # Do the path maximization.
    start = timer()
    E = path_max(P, method="MC", samples=s)
    elapsed = timer() - start
    info[nt]["DIST"] = E
    info[nt]["TIME"].append(elapsed)       
            
                        
# Save the info dict.
with open('{}/chol_paths.dill'.format(data_dest), 'wb') as handle:
    dill.dump(info, handle)  