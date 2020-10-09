#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kamburowski's bound.
"""

import dill, sys
from timeit import default_timer as timer
sys.path.append('../') 
from src import SDAG

chol_dag_path = '../graphs/cholesky_heft_accelerated'
nb = 128
n_tasks = [35, 220, 680, 1540]#, 2925, 4960, 7770, 11480]

for nt in n_tasks:
    print("\n{} TASKS".format(nt))
    # Load the DAG.
    with open('{}/nb{}/{}tasks.dill'.format(chol_dag_path, nb, nt), 'rb') as file:
        G = dill.load(file)
    H = SDAG(G)    
    # Get a reference solution for the longest path/makespan distribution using Monte Carlo sampling. 
    # (Only 100 samples so not great for larger DAGs but at least in the right ballpark.)
    mc = H.monte_carlo(samples=100)
    # PERT bound on the mean.
    pb = H.longest_path(pert_bound=True)[H.top_sort[-1].ID]
    # Kamburowski.
    lm, um, ls, us = H.kamburowski()
    
    print("---------------------------------")
    print("BEFORE RUNTIME")
    print("---------------------------------")    
    print("Reference solution: {}".format(mc))
    print("PERT-CPM bound on mean: {}".format(pb))
    print("Kamburowksi bounds on mean: ({}, {})".format(lm[H.top_sort[-1].ID], um[H.top_sort[-1].ID]))
    print("Kamburowksi bounds on variance: ({}, {})".format(ls[H.top_sort[-1].ID], us[H.top_sort[-1].ID]))