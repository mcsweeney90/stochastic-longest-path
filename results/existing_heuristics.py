#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimating the makespan/longest path distribution before runtime.
"""

import dill, pathlib, sys, os
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from scipy.stats import norm
from collections import defaultdict
sys.path.append('../') 
from src import RV, SDAG

data_dest = "data"

# =============================================================================
# Cholesky.
# =============================================================================

chol_dag_path = '../graphs/cholesky'
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]

info = {}
for nt in n_tasks:
    info[nt] = {}

for nt in n_tasks:
    with open('{}/nb128/{}tasks.dill'.format(chol_dag_path, nt), 'rb') as file:
        G = dill.load(file)
    H = SDAG(G)
    
    # PERT bound on the mean.
    pb = H.pert_cpm()[H.top_sort[-1].ID]
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
    
# Save the info dict.
with open('{}/chol_existing.dill'.format(data_dest), 'wb') as handle:
    dill.dump(info, handle)  
    
# =============================================================================
# STG.
# =============================================================================

stg_dag_path = '../graphs/STG'
info = {}
for dname in os.listdir(stg_dag_path):
    info[dname] = {}
timings = defaultdict(float)

for dname in os.listdir(stg_dag_path):    
    # print("Starting DAG {}...".format(i))
    with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
        H = dill.load(file)
        
    # PERT bound on the mean.
    start = timer()
    pb = H.pert_cpm()[H.top_sort[-1].ID]
    elapsed = timer() - start
    timings["PERT"] += elapsed
    info[dname]["PERT"] = pb
    
    # Kamburowski.
    start = timer()
    lm, um, ls, us = H.kamburowski()
    elapsed = timer() - start
    timings["KAMBUROWSKI"] += elapsed
    info[dname]["KML"] = lm[H.top_sort[-1].ID]
    info[dname]["KMU"] = um[H.top_sort[-1].ID]
    info[dname]["KVL"] = ls[H.top_sort[-1].ID]
    info[dname]["KVU"] = us[H.top_sort[-1].ID]
    
    # Sculli.
    start = timer()
    SL = H.sculli()
    elapsed = timer() - start
    timings["SCULLI"] += elapsed   
    sculli_forward = SL[H.top_sort[-1].ID]
    info[dname]["SCULLI"] = sculli_forward
    
    # CorLCA.
    start = timer()
    CL = H.corLCA()
    elapsed = timer() - start
    timings["CorLCA"] += elapsed
    corlca_forward = CL[H.top_sort[-1].ID]
    info[dname]["CorLCA"] = corlca_forward
        
# Save the info dict.
with open('{}/stg_existing.dill'.format(data_dest), 'wb') as handle:
    dill.dump(info, handle)  
with open('{}/stg_existing_timings.dill'.format(data_dest), 'wb') as handle:
    dill.dump(timings, handle) 