#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirical distribution of longest path.
"""

import dill, pathlib, sys
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from scipy.stats import norm
sys.path.append('../') 
from src import RV, SDAG

chol_dag_path = '../graphs/cholesky_heft_accelerated'
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]
samples = 100000

info = {}
for nt in n_tasks:
    info[nt] = {}
    info[nt]["NORMAL"] = []
    info[nt]["GAMMA"] = []
    
with open("empirical_dists_timing.txt", "w") as dest:
    print("---------------------------------", file=dest)
    print("NUMBER OF SAMPLES: {}".format(samples), file=dest)
    print("---------------------------------", file=dest)
    for nt in n_tasks:
        print("\nNUMBER OF TASKS: {}".format(nt), file=dest)
        with open('{}/nb128/{}tasks.dill'.format(chol_dag_path, nt), 'rb') as file:
            G = dill.load(file)
        H = SDAG(G)    
        for dist in ["NORMAL", "GAMMA"]:   
            start = timer()
            for i in range(samples):
                H.realize(dist=dist)
                Z = H.longest_path()
                lp = Z[H.top_sort[-1].ID]     # Assumes single exit task.
                info[nt][dist].append(lp)
            elapsed = timer() - start
            print("{} COSTS: {} minutes".format(dist, elapsed/60), file=dest)
            
with open('empirical_dists.dill', 'wb') as handle:
    dill.dump(info, handle) 
    
    