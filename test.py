#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing...
"""

import dill
import numpy as np
import networkx as nx
from timeit import default_timer as timer
# from src import RV, SDAG

chol_dag_path = 'graphs/cholesky'
n_tasks = [22100]#[35, 220, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]



s = 100000
for nt in n_tasks:
    with open('{}/{}.dill'.format(chol_dag_path, nt), 'rb') as file:
        G = dill.load(file)
        
    print("\n\n\n------------------------------------------------------------------")
    print("GRAPH SIZE: {}".format(G.size)) 
    print("------------------------------------------------------------------")
    
    # # CPM bound.
    # start = timer()
    # C = G.CPM()
    # elapsed = timer() - start
    # print("\nCPM bound on mean: {}".format(C[G.top_sort[-1]]))
    # print("Time: {}".format(elapsed))
    
    # # Kamburowski.
    # start = timer()
    # lm, um, ls, us = G.kamburowski()
    # elapsed = timer() - start
    # print("\nK. lower bound on mean: {}".format(lm[G.top_sort[-1]]))
    # print("K. upper bound on mean: {}".format(um[G.top_sort[-1]]))
    # print("K. lower bound on variance: {}".format(ls[G.top_sort[-1]]))
    # print("K. upper bound on variance: {}".format(us[G.top_sort[-1]]))
    # print("Time: {}".format(elapsed))
    
    # # Sculli.
    # start = timer()
    # SL = G.sculli()
    # elapsed = timer() - start
    # print("\nSculli's LP estimate: {}".format(SL[G.top_sort[-1]]))
    # print("Time: {}".format(elapsed))
    
    # # CorLCA.
    # start = timer()
    # CL = G.corLCA()
    # elapsed = timer() - start
    # print("\nCorLCA LP estimate: {}".format(CL[G.top_sort[-1]]))
    # print("Time: {}".format(elapsed))
    
    # MC.
    # start = timer()
    # E = G.monte_carlo(samples=s, dist="N")
    # elapsed = timer() - start
    # print("\nMC{} LP estimate: RV({}, {})".format(s, np.mean(E), np.var(E)))
    # print("Time: {}".format(elapsed))
    
    # Numpy MC.
    start = timer()
    E = G.np_mc(samples=s, dist="N")
    elapsed = timer() - start
    print("\nNPMC{} LP estimate: RV({}, {})".format(s, np.mean(E), np.var(E)))
    print("Time: {}".format(elapsed))
    