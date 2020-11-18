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
from src import RV, SDAG, Path

chol_dag_path = '../graphs/cholesky'
nb = 128
n_tasks = [35, 220]#, 680]#, 1540, 2925, 4960, 7770, 11480]

# =============================================================================
# Timings.
# =============================================================================


for nt in n_tasks:
    with open('{}/nb{}/{}tasks.dill'.format(chol_dag_path, nb, nt), 'rb') as file:
        G = dill.load(file)
    H = SDAG(G)
    
    print("\nNUMBER OF TASKS: {}".format(nt))
    
    start = timer()
    longest_paths = H.orig_dodin_longest_paths()
    elapsed = timer() - start
    print("Time to find longest paths: {}".format(elapsed))
    print(len(longest_paths))
        
    # start = timer()
    # mc30 = H.monte_carlo(samples=30)
    # elapsed = timer() - start
    # print("Time for MC30: {}".format(elapsed))
    
    # start = timer()
    # dom_path = longest_paths[0]
    # lp = longest_paths[0].length
    # for path in longest_paths[1:]:
    #     r = path.get_rho(dom_path)
    #     if path.length.mu > lp.mu:
    #         dom_path = path
    #     lp = lp.clark_max(path.length, rho=r)
    # elapsed = timer() - start
    # print(lp)
    # print("Time for maximization: {}".format(elapsed))
    
    # lp = CFP(H)
    # print(lp)
    
    # total, dis = 0, 0
    # for t in H.top_sort:
    #     for p in H.graph.predecessors(t):
    #         total += 1
    #         e = H.graph[p][t]['weight']
    #         try:
    #             m = e.mu
    #         except AttributeError:
    #             dis += 1
    # print("Edge ratio: {}".format(dis/total))
                
    
    # emp, paths = H.monte_carlo(samples=10, path_info=True)
    # print(len(paths))
    
    # start = timer()
    # # pert_est = H.CPM(variance=True)[H.top_sort[-1].ID]
    # intervals = H.bootstrap_confidence_intervals(samples=10, resamples=40000)
    # print(intervals)
    # elapsed = timer() - start
    
    # print("\n\n\n---------------------------------")
    # print("NUMBER OF TASKS: {}".format(nt)) 
    # print("---------------------------------")
    
    # print("\nREFERENCE SOLUTION: {}".format(before[nt]["MCN"][-1])) 
    
    # print("\nBOUNDS") 
    # print("PERT-CPM bound on mean: {}".format(before[nt]["PERT"]))
    # print("Kamburowski bounds on mean: ({}, {})".format(before[nt]["KML"], before[nt]["KMU"]))
    # print("Kamburowski bounds on variance: ({}, {})".format(before[nt]["KVL"], before[nt]["KVU"]))
    
    # print("\nAPPROXIMATIONS") 
    # # print("PERT estimate: {}".format(pert_est))
    # print("Sculli forward: {}".format(before[nt]["SCULLI"]))
    # print("Sculli backward: {}".format(before[nt]["SCULLI-R"]))
    # print("CorLCA forward: {}".format(before[nt]["CorLCA"]))
    # print("CorLCA backward: {}".format(before[nt]["CorLCA-R"]))    
        
    # # Realize half the tasks.    
    # # Z, fixed = H.partially_realize(fraction=0.5, percentile=0.9999999, return_info=True)
    # Z, fixed = H.partially_realize(fraction=0.5, percentile=None, return_info=True)
    
    # # Get a new estimate of the makespan distribution.
    # # TODO: really need a faster way to do this.
    # new_reals = []
    # for _ in range(100):
    #     H.realize(fixed=fixed)
    #     L = H.longest_path()
    #     lp = L[H.top_sort[-1].ID] 
    #     new_reals.append(lp)
    # H.reset(fixed=fixed)
    # new_mc = RV(np.mean(new_reals), np.var(new_reals))
        
    # # Updated Sculli.
    # # TODO: what about realized edges leading from boundary tasks?
    # new_sculli = None   
    # for t in H.top_sort:
    #     if t.ID not in fixed:
    #         continue
    #     if any(c.ID not in fixed for c in H.graph.successors(t)):
    #         nw = Z[t.ID] + SR[t.ID]
    #         if new_sculli is None:
    #             new_sculli = nw
    #         else:
    #             new_sculli = new_sculli.clark_max(nw, rho=0)
    
    # # Updated CorLCA.
    # # TODO: ditto above.
    # new_corlca, dom_term = None, None   
    # for t in H.top_sort:
    #     if t.ID not in fixed:
    #         continue
    #     if any(c.ID not in fixed for c in H.graph.successors(t)):
    #         nw = Z[t.ID] + CR[t.ID]
    #         if new_corlca is None:
    #             new_corlca = nw
    #             dom_term = t
    #         else:
    #             get_lca = nx.algorithms.tree_all_pairs_lowest_common_ancestor(backward_tree, pairs=[(dom_term.ID, t.ID)])
    #             lca = list(get_lca)[0][1] 
    #             r = BC[lca].var / (np.sqrt(BC[dom_term.ID].var) * np.sqrt(BC[t.ID].var))
    #             if nw.mu > new_corlca.mu:
    #                 dom_term = t
    #             new_corlca = new_corlca.clark_max(nw, rho=r)
    
    # # Correlation-based update rule. 
    # # TODO: maximization with or without correlations?
    # lp = CL[H.top_sort[-1].ID]
    # corr_update, dom_term = None, None
    # for t in H.top_sort:
    #     if t.ID not in fixed:
    #         continue
    #     if any(c.ID not in fixed for c in H.graph.successors(t)):
    #         # Get a new final makespan estimate...
    #         get_lca = nx.algorithms.tree_all_pairs_lowest_common_ancestor(forward_tree, pairs=[(H.top_sort[-1].ID, t.ID)])
    #         lca = list(get_lca)[0][1]
    #         rho = FC[lca].var / (np.sqrt(FC[H.top_sort[-1].ID].var) * np.sqrt(FC[t.ID].var))
    #         mu_add = ((rho * np.sqrt(lp.var)) / np.sqrt(CL[t.ID].var)) * (Z[t.ID] - CL[t.ID].mu)
    #         var_dash = (1 - rho*rho) * lp.var
    #         nw = RV(lp.mu + mu_add, var_dash)
    #         if dom_term is None:
    #             corr_update = nw
    #             dom_term = t
    #         else:
    #             get_lca = nx.algorithms.tree_all_pairs_lowest_common_ancestor(forward_tree, pairs=[(dom_term.ID, t.ID)])
    #             lca = list(get_lca)[0][1]
    #             r = FC[lca].var / (np.sqrt(FC[dom_term.ID].var) * np.sqrt(FC[t.ID].var))
    #             if nw.mu > corr_update.mu:
    #                 dom_term = t
    #             corr_update = corr_update.clark_max(nw, rho=r)
    
    # # Updated CorLCA.    
    # UF = H.update_corLCA(L=CL, correlation_tree=forward_tree, C=FC)
    # up_corlca = UF[H.top_sort[-1].ID]    
    
    # print("\nNumber of tasks: {}".format(nt))
    
    # print("---------------------------------")
    # print("BEFORE RUNTIME")
    # print("---------------------------------")    
    # print("MC-100: {}".format(mc))
    # print("PERT-CPM bound on mean: {}".format(pb))
    # print("Kamburowksi bounds on mean: ({}, {})".format(lm[H.top_sort[-1].ID], um[H.top_sort[-1].ID]))
    # print("Kamburowksi bounds on variance: ({}, {})".format(ls[H.top_sort[-1].ID], us[H.top_sort[-1].ID]))
    # print("Sculli forward: {}".format(sculli_forward))
    # print("Sculli backward: {}".format(sculli_backward))
    # print("CorLCA forward: {}".format(corlca_forward))
    # print("CorLCA backward: {}".format(corlca_backward))
    
    # print("\n---------------------------------")
    # print("HALFWAY THROUGH RUNTIME")
    # print("---------------------------------")
    # print("MC-100: {}".format(new_mc))
    # print("New Sculli: {}".format(new_sculli))
    # print("New CorLCA: {}".format(new_corlca))
    # print("Correlation-based update: {}".format(corr_update))
    # print("Update CorLCA: {}".format(up_corlca))
    
    # print("\nTime taken: {}".format(elapsed))
    