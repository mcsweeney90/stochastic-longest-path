#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timing script.
"""

import dill, pathlib, sys, os
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from scipy.stats import norm
sys.path.append('../') 
from src import RV, SDAG

# =============================================================================
# Cholesky.
# =============================================================================

# chol_dag_path = '../graphs/cholesky'
# nb = 128
# n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]

# with open("chol_timings.txt", "w") as dest:
#     for nt in n_tasks:
#         print("\n{} TASKS".format(nt), file=dest)
#         # Loading DAG.
#         start = timer()
#         with open('{}/nb{}/{}tasks.dill'.format(chol_dag_path, nb, nt), 'rb') as file:
#             G = dill.load(file)
#         H = SDAG(G)
#         elapsed = timer() - start
#         print("Loading DAG: {} seconds".format(elapsed), file=dest)
        
#         # PERT bound.
#         start = timer()
#         pb = H.pert_cpm()
#         elapsed = timer() - start
#         print("PERT bound: {} seconds".format(elapsed), file=dest)
        
#         # Kamburowski.
#         start = timer()
#         lm, um, ls, us = H.kamburowski()
#         elapsed = timer() - start
#         print("Kamburowski bound: {} seconds".format(elapsed), file=dest)
        
#         # Sculli.
#         start = timer()
#         SL = H.sculli()
#         elapsed = timer() - start
#         print("Sculli: {} seconds".format(elapsed), file=dest)
        
#         # CorLCA.
#         start = timer()
#         CL = H.corLCA()
#         elapsed = timer() - start
#         print("CorLCA: {} seconds".format(elapsed), file=dest)
            
#         # Monte Carlo.
#         for s in [10]:
#             start = timer()
#             mc = H.monte_carlo(samples=s)
#             elapsed = timer() - start
#             print("Monte Carlo ({} runs): {} seconds".format(s, elapsed), file=dest)
            
# =============================================================================
# STG.
# =============================================================================

stg_dag_path = '../graphs/STG'

with open("stg_timing.txt", "w") as dest:
    pb_time, k_time, s_time, c_time, mc_time = 0.0, 0.0, 0.0, 0.0, 0.0
    for i, dname in enumerate(os.listdir(stg_dag_path)):    
        print("Starting DAG {}...".format(i))
        # Loading DAG.
        start = timer()
        with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
            H = dill.load(file)
        elapsed = timer() - start
        print("Loading DAG: {} seconds".format(elapsed), file=dest)
        
        # PERT bound.
        start = timer()
        pb = H.pert_cpm()
        elapsed = timer() - start
        pb_time += elapsed
        
        # Kamburowski.
        start = timer()
        lm, um, ls, us = H.kamburowski()
        elapsed = timer() - start
        k_time += elapsed
        
        # Sculli.
        start = timer()
        SL = H.sculli()
        elapsed = timer() - start
        s_time += elapsed
        
        # CorLCA.
        start = timer()
        CL = H.corLCA()
        elapsed = timer() - start
        c_time += elapsed
            
        # Monte Carlo.
        for s in [10]:
            start = timer()
            mc = H.monte_carlo(samples=s)
            elapsed = timer() - start
            mc_time += elapsed
    
    print("AVG. # MC SAMPLES THAT CAN BE DONE IN SAME TIME", file=dest)
    pbm = (pb_time / mc_time) * 10
    print("\nCPM BOUND: {}".format(pbm), file=dest)
    km = (k_time / mc_time) * 10
    print("\nKAMBUROWSKI'S BOUNDS: {}".format(km), file=dest)
    sm = (s_time / mc_time) * 10
    print("\nSCULLI'S METHOD: {}".format(sm), file=dest)
    cm = (c_time / mc_time) * 10
    print("\nCorLCA: {}".format(cm), file=dest)
    
        
        