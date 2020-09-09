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
from src import SDAG

chol_dag_path = '../graphs/cholesky_heft_accelerated'

nb = 128

n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]

for nt in n_tasks:
    start = timer()
    with open('{}/nb{}/{}tasks.dill'.format(chol_dag_path, nb, nt), 'rb') as file:
        G = dill.load(file)
    H = SDAG(G)
    s = H.sculli()
    lp = H.longest_path(expected=True)
    m = H.monte_carlo()
    elapsed = timer() - start
    print("\nNumber of tasks: {}".format(nt))
    print("MC estimate: {}".format(m))
    print("PERT bound on mean: {}".format(lp))
    print("Sculli's estimate: {}".format(s))
    print("Time taken: {}".format(elapsed))
    