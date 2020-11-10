#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirical distribution of longest path.
"""

import dill, pathlib, sys, os
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from scipy.stats import norm
sys.path.append('../') 
from src import RV, SDAG

data_dest = "data"

# =============================================================================
# Cholesky.
# =============================================================================

# chol_dag_path = '../graphs/cholesky'
# n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]
# samples = 100000

# info = {}
# for nt in n_tasks:
#     info[nt] = {}
#     info[nt]["NORMAL"] = []
#     info[nt]["GAMMA"] = []
    
# with open("chol_empirical_timing.txt", "w") as dest:
#     print("---------------------------------", file=dest)
#     print("NUMBER OF SAMPLES: {}".format(samples), file=dest)
#     print("---------------------------------", file=dest)
#     for nt in n_tasks:
#         print("\nNUMBER OF TASKS: {}".format(nt), file=dest)
#         with open('{}/nb128/{}tasks.dill'.format(chol_dag_path, nt), 'rb') as file:
#             G = dill.load(file)
#         H = SDAG(G)    
#         for dist in ["NORMAL", "GAMMA"]:   
#             start = timer()
#             for i in range(samples):
#                 H.realize(dist=dist)
#                 Z = H.longest_path()
#                 lp = Z[H.top_sort[-1].ID]     # Assumes single exit task.
#                 info[nt][dist].append(lp)
#             elapsed = timer() - start
#             print("{} WEIGHTS: {} minutes".format(dist, elapsed/60), file=dest)
            
# with open('{}/chol_empirical.dill'.format(data_dest), 'wb') as handle:
#     dill.dump(info, handle) 

# =============================================================================
# STG.
# =============================================================================

stg_dag_path = '../graphs/STG'
samples = 10000

info = {}
for dname in os.listdir(stg_dag_path):
    info[dname] = {}
    info[dname]["NORMAL"] = []
    info[dname]["GAMMA"] = []

with open("stg_empirical_timing.txt", "w") as dest:
    start = timer()
    for dname in os.listdir(stg_dag_path):    
        # print("Starting DAG {}...".format(i))
        with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
            G = dill.load(file)
        for dist in ["NORMAL", "GAMMA"]: 
            mc = G.monte_carlo(samples=samples)
            mu = np.mean(mc)
            var = np.var(mc)
            info[dname][dist].append(mu)
            info[dname][dist].append(var)
    elapsed = timer() - start
    print("TIME TAKEN: {} minutes".format(elapsed/60), file=dest)

with open('{}/stg_empirical.dill'.format(data_dest), 'wb') as handle:
    dill.dump(info, handle)       
    

