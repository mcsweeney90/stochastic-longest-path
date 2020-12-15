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
from src import RV, SDAG, summary_statistics

data_dest = "data"

# =============================================================================
# Cholesky.
# =============================================================================

# chol_dag_path = '../graphs/cholesky'
# n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]

# info, timings = {}, {}

# for nt in n_tasks:
#     with open('{}/nb128/{}tasks.dill'.format(chol_dag_path, nt), 'rb') as file:
#         G = dill.load(file)
#     H = SDAG(G)
#     timings[nt], info[nt] = {}, {}
    
#     for dist in ["NORMAL", "GAMMA", "UNIFORM"]:
#         timings[nt][dist], info[nt][dist] = {}, {}
#         for s in [10, 30, 100, 1000]:
#             timings[nt][dist][s], info[nt][dist][s] = {}, {}
            
#             # Get the reference solution.
#             start = timer()
#             full_emp = H.monte_carlo(samples=s, dist=dist)
#             elapsed = timer() - start
#             info[nt][dist][s]["FULL"] = full_emp 
#             timings[nt][dist][s]["FULL"] = elapsed 
            
#             for W in ["MEAN", "UCB"]:
#                 for L in [0, 0.6, 0.8]:  
#                     start = timer()
#                     C = H.get_critical_subgraph(node_limit=L, weights=W)
#                     get_subgraph = timer() - start
#                     emp = C.monte_carlo(samples=s, dist=dist)
#                     elapsed = timer() - start
#                     info[nt][dist][s][(W, L)] = emp
#                     timings[nt][dist][s][(W, L)] = (elapsed, get_subgraph) 
                        
# # Save the info dict.
# with open('{}/chol_prune.dill'.format(data_dest), 'wb') as handle:
#     dill.dump(info, handle)  
# with open('{}/chol_prune_timings.dill'.format(data_dest), 'wb') as handle:
#     dill.dump(timings, handle) 