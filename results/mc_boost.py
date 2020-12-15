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

chol_dag_path = '../graphs/cholesky'
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]

info = {}
for nt in n_tasks:
    print("\nNUMBER OF TASKS: {}".format(nt))
    with open('{}/nb128/{}tasks.dill'.format(chol_dag_path, nt), 'rb') as file:
        G = dill.load(file)
    H = SDAG(G)
    info[nt] = {}
    
    for dist in ["NORMAL", "GAMMA", "UNIFORM"]:
        print(dist)
        info[nt][dist] = {}
        for s in [30, 100]:
            # Do the MC realizations and record the observed longest paths.
            start = timer()
            emp_dist, P = H.monte_carlo(samples=s, dist=dist, return_paths=True)
            record = timer() - start
            info[nt][dist]["MC{}".format(s)] = emp_dist
                                
            # Sculli and CorLCA maximization variants.
            for mthd in ["S", "C"]:
                start = timer()
                L = path_max(P, method=mthd)
                elapsed = timer() - start
                info[nt][dist]["MC{}-{}".format(s, mthd)] = L 
                info[nt][dist]["MC{}-{} TIME".format(s, mthd)] = [record, elapsed] 
            
            # Monte Carlo.
            start = timer()      
            # Compute covariance matrix.
            cov = []
            for i, pth in enumerate(P):
                row = []
                # Copy already computed covariances.
                row = [cov[j][i] for j in range(i)]
                # Add diagonal - just the variance.
                row.append(pth.length.var)
                # Compute covariance with other paths.
                for pt in P[i + 1:]: 
                    rho = pth.get_rho(pt)
                    cv = rho * np.sqrt(pth.length.var) * np.sqrt(pt.length.var)
                    row.append(cv)
                cov.append(row)  
            cov_time = timer() - start
            # Construct vector of means.
            means = [pth.length.mu for pth in P] 
            # Generate the path length realizations.
            N = np.random.default_rng().multivariate_normal(means, cov, 100*s)        
            # Compute the maximums.
            mx = np.amax(N, axis=1)  
            E = list(mx)
            elapsed = timer() - start
            info[nt][dist]["MC{}-MC".format(s)] = E   
            info[nt][dist]["MC{}-MC TIME".format(s)] = [record, cov_time, elapsed]
    
# Save the info dict.
with open('{}/chol_boost.dill'.format(data_dest), 'wb') as handle:
    dill.dump(info, handle)  

# =============================================================================
# STG.
# =============================================================================

# stg_dag_path = '../graphs/STG'
# info = {}
# for dname in os.listdir(stg_dag_path):
#     info[dname] = {}
# timings = defaultdict(float)

# for dname in os.listdir(stg_dag_path):  
#     with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
#         H = dill.load(file)
        
#     for s in [30, 100]:
#         # Do the MC realizations and record the observed longest paths.
#         start = timer()
#         emp_dist, P = H.monte_carlo(samples=s, return_paths=True)
#         elapsed = timer() - start
#         timings["MC + RECORD PATHS"] += elapsed    
                
#         # Empirical mean and variance.
#         mu = np.mean(emp_dist)
#         var = np.var(emp_dist)
#         info[dname]["MC{}".format(s)] = RV(mu, var)
        
#         # Sculli and CorLCA maximization variants.
#         for mthd in ["SCULLI", "CorLCA"]:
#             start = timer()
#             L = path_max(P, method=mthd)
#             elapsed = timer() - start
#             timings["MC{}-{}".format(s, mthd)] += elapsed
#             info[dname]["MC{}-{}".format(s, mthd)] = L
        
#         # Monte Carlo.
#         start = timer()
#         E = path_max(P, method="MC", samples=1000*s)
#         m, v = np.mean(E), np.var(E)
#         elapsed = timer() - start
#         timings["MC{}-MC".format(s)] += elapsed
#         info[dname]["MC{}-MC".format(s)] = RV(m, v) 
    
        
# # Save the info dict.
# with open('{}/stg_boost.dill'.format(data_dest), 'wb') as handle:
#     dill.dump(info, handle)  
# with open('{}/stg_boost_timings.dill'.format(data_dest), 'wb') as handle:
#     dill.dump(timings, handle) 