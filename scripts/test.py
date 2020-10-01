#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing...
"""

import dill, pathlib, sys
import numpy as np
import networkx as nx
from timeit import default_timer as timer
sys.path.append('../') 
from src import RV, SDAG

chol_dag_path = '../graphs/cholesky_heft_accelerated'

nb = 128

n_tasks = [1540]#[35, 220, 680, 1540]#, 2925, 4960, 7770, 11480]

for nt in n_tasks:
    with open('{}/nb{}/{}tasks.dill'.format(chol_dag_path, nb, nt), 'rb') as file:
        G = dill.load(file)
    H = SDAG(G)
    # s = H.sculli()
    pb = H.longest_path(pert_bound=True)
    start = timer()
    m = H.monte_carlo(samples=100)
    # c = H.corLCA()
    # 
    # last = H.size // 2
    # H.realize(last=last)
    # lp = H.longest_path()
    
    # H.realize(first=last)
    # lp2 = H.longest_path()
    
    elapsed = timer() - start
    print("\nNumber of tasks: {}".format(nt))
    print("MC estimate: {}".format(m))
    print("PERT bound on mean: {}".format(pb))
    # print("Sculli's estimate: {}".format(s))
    # print("CorLCA estimate: {}".format(c))
    # print("Updated PERT bound: {}".format(lp))
    # print("Realized longest path: {}".format(lp2))
    print("Time taken: {}".format(elapsed))
    