#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path-centric heuristics.
"""

import dill, pathlib, sys, os
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from scipy.stats import norm
from collections import defaultdict
sys.path.append('../') 
from src import RV, SDAG

data_dest = "data"
samples = 30

# =============================================================================
# Cholesky.
# =============================================================================

# chol_dag_path = '../graphs/cholesky'
# n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]

# info = {}
# for nt in n_tasks:
#     info[nt] = {}
# timings = defaultdict(list)

# for nt in n_tasks:
#     with open('{}/nb128/{}tasks.dill'.format(chol_dag_path, nt), 'rb') as file:
#         G = dill.load(file)
#     H = SDAG(G)
        
#     # Get longest paths.
#     start = timer()
#     longest_paths, lp_candidates = [], []  
#     unique = set()
#     for _ in range(samples):
#         H.realize()
#         lp, P = H.real_longest_path(return_path=True)
#         longest_paths.append(lp)
#         check = P.get_rep()
#         if check not in unique:
#             lp_candidates.append(P)
#             unique.add(check)                
#     H.reset()
#     elapsed = timer() - start
#     timings["FIND PATHS"].append(elapsed)
    
#     # Simple Monte Carlo mean and variance.
#     mu = np.mean(longest_paths)
#     var = np.var(longest_paths)
#     info[nt]["MC30"] = RV(mu, var)
    
#     # Sculli's method on paths. 
#     start = timer()
#     L = lp_candidates[0].length
#     for path in lp_candidates[1:]:
#         L = L.clark_max(path.length)
#     elapsed = timer() - start
#     timings["MC30-SCULLI"].append(elapsed)
#     info[nt]["MC30-SCULLI"] = L
    
#     # CorLCA analogue on paths.
#     start = timer()
#     dom_path = lp_candidates[0]
#     C = lp_candidates[0].length
#     for path in lp_candidates[1:]:
#         r = path.get_rho(dom_path)
#         if path.length.mu > C.mu:
#             dom_path = path
#         C = C.clark_max(path.length, rho=r)
#     elapsed = timer() - start
#     timings["MC30-CorLCA"].append(elapsed)
#     info[nt]["MC30-CorLCA"] = C
    
# # Save the info dict.
# with open('{}/chol_path.dill'.format(data_dest), 'wb') as handle:
#     dill.dump(info, handle)  
# with open('{}/chol_path_timings.dill'.format(data_dest), 'wb') as handle:
#     dill.dump(timings, handle) 

# =============================================================================
# STG.
# =============================================================================

stg_dag_path = '../graphs/STG'
info = {}
for dname in os.listdir(stg_dag_path):
    info[dname] = {}
timings = defaultdict(float)

for dname in os.listdir(stg_dag_path):  
    with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
        H = dill.load(file)
    
    start = timer()
    longest_paths = H.dodin_longest_paths(epsilon=0.1)
    # CL = H.corLCA()
    elapsed = timer() - start
    print(elapsed)
    
    
    
        
    # # Get longest paths.
    # start = timer()
    # longest_paths, lp_candidates = [], []  
    # unique = set()
    # for _ in range(samples):
    #     H.realize()
    #     lp, P = H.real_longest_path(return_path=True)
    #     longest_paths.append(lp)
    #     check = P.get_rep()
    #     if check not in unique:
    #         lp_candidates.append(P)
    #         unique.add(check)                
    # H.reset()
    # elapsed = timer() - start
    # timings["FIND PATHS"] += elapsed    
    
    # # Simple Monte Carlo mean and variance.
    # mu = np.mean(longest_paths)
    # var = np.var(longest_paths)
    # info[dname]["MC30"] = RV(mu, var)
    
    # # Sculli's method on paths. 
    # start = timer()
    # L = lp_candidates[0].length
    # for path in lp_candidates[1:]:
    #     L = L.clark_max(path.length)
    # elapsed = timer() - start
    # timings["MC30-SCULLI"] += elapsed
    # info[dname]["MC30-SCULLI"] = L
    
    # # CorLCA analogue on paths.
    # start = timer()
    # dom_path = lp_candidates[0]
    # C = lp_candidates[0].length
    # for path in lp_candidates[1:]:
    #     r = path.get_rho(dom_path)
    #     if path.length.mu > C.mu:
    #         dom_path = path
    #     C = C.clark_max(path.length, rho=r)
    # elapsed = timer() - start
    # timings["MC30-CorLCA"] += elapsed
    # info[dname]["MC30-CorLCA"] = C
        
# # Save the info dict.
# with open('{}/stg_path.dill'.format(data_dest), 'wb') as handle:
#     dill.dump(info, handle)  
# with open('{}/stg_path_timings.dill'.format(data_dest), 'wb') as handle:
#     dill.dump(timings, handle) 