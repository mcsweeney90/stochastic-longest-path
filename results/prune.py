#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduced MC methods.
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
    
# =============================================================================
# STG.
# =============================================================================

stg_dag_path = '../graphs/STG'
s = 10000

info = {}

for dname in os.listdir(stg_dag_path):  
    info[dname] = {}
    with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
        G = dill.load(file)
        
    for dist in ["NORMAL", "GAMMA", "UNIFORM"]:
        info[dname][dist] = {}
            
        # Get the reference solution.
        start = timer()
        full_emp = G.monte_carlo(samples=s, dist=dist)
        elapsed = timer() - start
        ref_sum_stats = summary_statistics(full_emp) 
        ref_sum_stats["TIME"] = elapsed
        info[dname][dist]["FULL"] = ref_sum_stats
        
    for W in ["MEAN", "UCB"]:
        for F in [0.5, 0.7, 0.9]:
            start = timer()
            C = G.get_critical_subgraph(f=F, weights=W) 
            get_subgraph = timer() - start
            
            for dist in ["NORMAL", "GAMMA", "UNIFORM"]:
                start = timer()
                emp = C.monte_carlo(samples=s, dist=dist)
                elapsed = timer() - start
                sum_stats = summary_statistics(emp)
                sum_stats["TIME"] = (elapsed, get_subgraph)
                # Need to compute KS statistic here since uses full distributions.
                ks, p = ks_2samp(emp, full_emp)
                sum_stats["KS"] = (ks, p)                
                info[dname][dist][(W, F)] = sum_stats        
        
with open('{}/stg_prune.dill'.format(data_dest), 'wb') as handle:
    dill.dump(info, handle) 
    
