#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small-scale testing of methods for estimating the makespan/longest path distribution before runtime.
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

info = {}
for nt in n_tasks:
    info[nt] = {}

for nt in n_tasks:
    with open('{}/nb128/{}tasks.dill'.format(chol_dag_path, nt), 'rb') as file:
        G = dill.load(file)
    H = SDAG(G)
    
    # PERT bound on the mean.
    pb = H.longest_path(pert_bound=True)[H.top_sort[-1].ID]
    info[nt]["PERT"] = pb
    # Kamburowski.
    lm, um, ls, us = H.kamburowski()
    info[nt]["KML"] = lm[H.top_sort[-1].ID]
    info[nt]["KMU"] = um[H.top_sort[-1].ID]
    info[nt]["KVL"] = ls[H.top_sort[-1].ID]
    info[nt]["KVU"] = us[H.top_sort[-1].ID]
    # Sculli.
    SL = H.sculli()
    SR = H.sculli(remaining=True)    
    sculli_forward = SL[H.top_sort[-1].ID]
    sculli_backward = SR[H.top_sort[0].ID] + H.top_sort[0]
    info[nt]["SCULLI"] = sculli_forward
    info[nt]["SCULLI-R"] = sculli_backward
    # CorLCA.
    CL = H.corLCA()
    CR = H.corLCA(remaining=True)
    corlca_forward = CL[H.top_sort[-1].ID]
    corlca_backward = CR[H.top_sort[0].ID] + H.top_sort[0]
    info[nt]["CorLCA"] = corlca_forward
    info[nt]["CorLCA-R"] = corlca_backward
    # Monte Carlo.
    # Normal costs.
    mcn = H.monte_carlo(samples=100) 
    info[nt]["MCN"] = mcn
    # Gamma costs.
    mcg = H.monte_carlo(samples=100, dist="GAMMA") 
    info[nt]["MCG"] = mcg   
    
# Save the info dict.
with open('before_runtime.dill', 'wb') as handle:
    dill.dump(info, handle)  
    
    
