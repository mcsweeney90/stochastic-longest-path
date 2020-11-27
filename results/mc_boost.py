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
from src import RV, SDAG, path_max

data_dest = "data"

# =============================================================================
# Cholesky.
# =============================================================================

# chol_dag_path = '../graphs/cholesky'
# n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]

# info = {}
# for nt in n_tasks:
#     info[nt] = {}
# timings = {}

# for nt in n_tasks:
#     with open('{}/nb128/{}tasks.dill'.format(chol_dag_path, nt), 'rb') as file:
#         G = dill.load(file)
#     H = SDAG(G)
#     timings[nt] = {}
    
#     for s in [30, 100]:
#         # Do the MC realizations and record the observed longest paths.
#         start = timer()
#         emp_dist, P = H.monte_carlo(samples=s, return_paths=True)
#         elapsed = timer() - start
#         timings[nt]["MC + RECORD PATHS"] = elapsed    
                
#         # Empirical mean and variance.
#         mu = np.mean(emp_dist)
#         var = np.var(emp_dist)
#         info[nt]["MC{}".format(s)] = RV(mu, var)
        
#         # Sculli and CorLCA maximization variants.
#         for mthd in ["SCULLI", "CorLCA"]:
#             start = timer()
#             L = path_max(P, method=mthd)
#             elapsed = timer() - start
#             timings[nt]["MC{}-{}".format(s, mthd)] = elapsed
#             info[nt]["MC{}-{}".format(s, mthd)] = L
        
#         # Monte Carlo.
#         start = timer()
#         E = path_max(P, method="MC", samples=1000*s)
#         m, v = np.mean(E), np.var(E)
#         elapsed = timer() - start
#         timings[nt]["MC{}-MC".format(s)] = elapsed
#         info[nt]["MC{}-MC".format(s)] = RV(m, v)        
    
# # Save the info dict.
# with open('{}/chol_boost.dill'.format(data_dest), 'wb') as handle:
#     dill.dump(info, handle)  
# with open('{}/chol_boost_timings.dill'.format(data_dest), 'wb') as handle:
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
        
    for s in [30, 100]:
        # Do the MC realizations and record the observed longest paths.
        start = timer()
        emp_dist, P = H.monte_carlo(samples=s, return_paths=True)
        elapsed = timer() - start
        timings["MC + RECORD PATHS"] += elapsed    
                
        # Empirical mean and variance.
        mu = np.mean(emp_dist)
        var = np.var(emp_dist)
        info[dname]["MC{}".format(s)] = RV(mu, var)
        
        # Sculli and CorLCA maximization variants.
        for mthd in ["SCULLI", "CorLCA"]:
            start = timer()
            L = path_max(P, method=mthd)
            elapsed = timer() - start
            timings["MC{}-{}".format(s, mthd)] += elapsed
            info[dname]["MC{}-{}".format(s, mthd)] = L
        
        # Monte Carlo.
        start = timer()
        E = path_max(P, method="MC", samples=1000*s)
        m, v = np.mean(E), np.var(E)
        elapsed = timer() - start
        timings["MC{}-MC".format(s)] += elapsed
        info[dname]["MC{}-MC".format(s)] = RV(m, v) 
    
        
# Save the info dict.
with open('{}/stg_boost.dill'.format(data_dest), 'wb') as handle:
    dill.dump(info, handle)  
with open('{}/stg_boost_timings.dill'.format(data_dest), 'wb') as handle:
    dill.dump(timings, handle) 