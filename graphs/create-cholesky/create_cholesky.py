#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create stochastic schedule DAGs for Cholesky.
"""

import dill, pathlib, sys
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from Simulator import Platform
sys.path.append('../../') 
from src import RV, SDAG

# Destinations etc.
platform = Platform(7, 1, name="Single_GPU")
chol_dag_path = 'task-graphs'
nb, adt = 128, "no_adt"
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480, 16215, 22100]

chol_dag_dest = '../cholesky'
pathlib.Path(chol_dag_dest).mkdir(parents=True, exist_ok=True)
# Load costs (for variances):
with open('skylake_V100_samples/{}_nb{}.dill'.format(adt, nb), 'rb') as file:
    comp_data, comm_data = dill.load(file)
            
variances = {}        
for kernel in ["GEMM", "POTRF", "SYRK", "TRSM"]:
    variances[kernel] = {}
    variances[kernel]["C"] = np.var(comp_data["C"][kernel])
    variances[kernel]["G"] = np.var(comp_data["G"][kernel])
    variances[kernel]["CC"] = np.var(comm_data["CC"][kernel])
    variances[kernel]["CG"] = np.var(comm_data["CG"][kernel])
    variances[kernel]["GC"] = np.var(comm_data["GC"][kernel])
    variances[kernel]["GG"] = np.var(comm_data["GG"][kernel])            
    
for nt in n_tasks:
    print("nt = {}".format(nt))
    with open('{}/nb{}/{}/{}tasks.dill'.format(chol_dag_path, nb, adt, nt), 'rb') as file:
        dag = dill.load(file) 
        
    G = nx.DiGraph()
        
    # Find HEFT schedule.
    priority_list = dag.sort_by_upward_rank(platform)
    for t in priority_list:    
        worker_finish_times = list(w.earliest_finish_time(t, dag, platform) for w in platform.workers)
        min_val = min(worker_finish_times, key=lambda w:w[0]) 
        min_worker = worker_finish_times.index(min_val)   
        ft, idx = min_val
        platform.workers[min_worker].schedule_task(t, finish_time=ft, load_idx=idx)
        # Create the corresponding node in the schedule DAG.
        mu = t.comp_costs["C"] if min_worker < platform.n_CPUs else t.comp_costs["G"]
        var = variances[t.type]["C"] if min_worker < platform.n_CPUs else variances[t.type]["G"]
        w = RV(mu, var)
        G.add_node(t.ID)
        G.nodes[t.ID]['weight'] = w
    
    # Add the edges and their weights. 
    for t in dag.top_sort:
        for s in list(dag.graph.successors(t)):
            # Add edge.
            G.add_edge(t.ID, s.ID)
            # Add the costs.
            if (t.where_scheduled == s.where_scheduled) or (platform.workers[t.where_scheduled].type == "C" and platform.workers[s.where_scheduled].type == "C"):
                w = 0.0
            else:
                source_type = platform.workers[t.where_scheduled].type
                target_type = platform.workers[s.where_scheduled].type 
                mu = t.comm_costs[source_type + target_type][s.ID]
                var = variances[s.type][source_type + target_type]
                w = RV(mu, var) 
            G[t.ID][s.ID]['weight'] = w
    
    # Add transitive edges.
    for p in platform.workers:
        if p.idle:
            continue
        elif len(p.load) == 1:
            continue
        for i, t in enumerate(p.load[:-1]):
            s = p.load[i + 1]
            if not G.has_edge(t[0], s[0]):
                G.add_edge(t[0], s[0])
                G[t[0]][s[0]]['weight'] = 0.0
                                    
    # Reset DAG and platform if necessary.
    dag.reset()
    platform.reset() 
    
    # Convert to SDAG.
    H = SDAG(G)                
    # Save.
    with open('{}/{}.dill'.format(chol_dag_dest, nt), 'wb') as handle:
        dill.dump(H, handle)
                        
                    
                    
                
                
            

