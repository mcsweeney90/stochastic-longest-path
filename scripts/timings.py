#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timing script.
"""

import dill, pathlib, sys
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from scipy.stats import norm
sys.path.append('../') 
from src import RV, SDAG

chol_dag_path = '../graphs/cholesky_heft_accelerated'
nb = 128
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]

for nt in n_tasks:
    print("\n{} TASKS".format(nt))
    # Loading DAG.
    start = timer()
    with open('{}/nb{}/{}tasks.dill'.format(chol_dag_path, nb, nt), 'rb') as file:
        G = dill.load(file)
    H = SDAG(G)
    elapsed = timer() - start
    print("Loading DAG: {} seconds".format(elapsed))
    
    # # PERT bound.
    # start = timer()
    # pb = H.longest_path(pert_bound=True)[H.top_sort[-1].ID]
    # elapsed = timer() - start
    # print("PERT bound: {} seconds".format(elapsed))
    
    # # Kamburowski.
    # start = timer()
    # lm, um, ls, us = H.kamburowski()
    # elapsed = timer() - start
    # print("Kamburowski bound: {} seconds".format(elapsed))
    
    # # Sculli.
    # start = timer()
    # SL = H.sculli()
    # elapsed = timer() - start
    # print("Sculli: {} seconds".format(elapsed))
    
    # CorLCA.
    # start = timer()
    # CL, forward_tree, FC = H.corLCA(return_correlation_tree=True)
    # elapsed = timer() - start
    # print("CorLCA: {} seconds".format(elapsed))
    
    # CorLCA-lite. TODO: still needs to be looked at again.
    # start = timer()
    # L = H.corLCA_lite()
    # elapsed = timer() - start
    # print("CorLCA lite: {} seconds".format(elapsed))
    
    # Monte Carlo.
    start = timer()
    mc = H.monte_carlo(samples=10)
    elapsed = timer() - start
    print("Monte Carlo (10 runs): {} seconds".format(elapsed))
    print("Estimated time for 10^5 runs: {} hours".format(elapsed * 10000 / 3600))