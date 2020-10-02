#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing...
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
n_tasks = [220]#[35, 220, 680, 1540]#, 2925, 4960, 7770, 11480]

for nt in n_tasks:
    with open('{}/nb{}/{}tasks.dill'.format(chol_dag_path, nb, nt), 'rb') as file:
        G = dill.load(file)
    H = SDAG(G)
    
    start = timer()
    
    # Get a reference solution (may need more samples of course).
    m = H.monte_carlo(samples=100)
    # Sculli.
    SL = H.sculli()
    SR = H.sculli(remaining=True)    
    sculli_forward = SL[H.top_sort[-1].ID]
    sculli_backward = SR[H.top_sort[0].ID] + H.top_sort[0]
    # CorLCA.
    CL = H.corLCA()
    CR = H.corLCA(remaining=True)
    corlca_forward = CL[H.top_sort[-1].ID]
    corlca_backward = CR[H.top_sort[0].ID] + H.top_sort[0]
    
    # Realize half the tasks.
    mid = H.size // 2
    H.realize(last=mid, percentile=0.99999)
    
    RZ = H.real_longest_path()
    
    new_scullis = []
    for t in H.top_sort:
        if t.realization is None:
            break
        if all(c.realization is None for c in H.graph.successors(t)):
            nw = RZ[t.ID] + SR[t.ID]
            new_scullis.append(nw)
    print(new_scullis)
    
    # Get a new estimate of the makespan distribution.
    # TODO: inefficient to do it this way, change real_longest_path function.
    reals = []
    for _ in range(100):
        H.realize(first=mid+1)
        Z = H.real_longest_path()
        lp = Z[H.top_sort[-1].ID]     # Assumes single exit task.
        reals.append(lp)
    updated_m = RV(np.mean(reals), np.var(reals))
    
    
    
    
    
    
    
    elapsed = timer() - start
    print("\nNumber of tasks: {}".format(nt))
    
    print("\n---------------------------------")
    print("BEFORE RUNTIME")
    print("---------------------------------")    
    print("MC-100: {}".format(m))
    print("Sculli forward: {}".format(sculli_forward))
    print("Sculli backward: {}".format(sculli_backward))
    print("CorLCA forward: {}".format(corlca_forward))
    print("CorLCA backward: {}".format(corlca_backward))
    
    print("\n---------------------------------")
    print("HALFWAY THROUGH RUNTIME")
    print("---------------------------------")
    print("MC-100: {}".format(updated_m))
    
    
    print("\nTime taken: {}".format(elapsed))
    