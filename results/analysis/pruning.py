#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph pruning MC.
"""

import dill, pathlib, sys
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import ks_2samp
sys.path.append('../../') # Needed for src apparently...

####################################################################################################

# Set some parameters for plots.
# See here: http://www.futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold' 
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.titlepad'] = 0
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

####################################################################################################

# Destinations to save summaries and generated plots.
summary_path = "../summaries/pruning"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "../plots/pruning"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

# Cholesky.
nb = 128
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]
with open('../data/chol_prune.dill', 'rb') as file:
    chol_prune = dill.load(file) 
with open('../data/chol_prune_timings.dill', 'rb') as file:
    chol_prune_timings = dill.load(file) 
    
# =============================================================================
# Cholesky.
# =============================================================================

# # Full summary.
# for dist, s in it.product(["NORMAL", "GAMMA", "UNIFORM"], [10, 30, 100, 1000]):
#     with open("{}/chol_{}_{}SAMPLES.txt".format(summary_path, dist, s), "w") as dest:
#         print("QUALITY OF PRUNED MC SOLUTIONS RELATIVE TO FULL MC SOLUTIONS.", file=dest)
#         print("WEIGHT DISTRIBUTIONS: {}.".format(dist), file=dest)
#         print("NUMBER OF SAMPLES: {}.".format(s), file=dest)
#         for nt in n_tasks: 
#             print("\n\n\n---------------------------------", file=dest)
#             print("NUMBER OF TASKS: {}".format(nt), file=dest) 
#             print("---------------------------------", file=dest)
            
#             print("------------------------------------------------------", file=dest)
                
#             ref_dist = chol_prune[nt][dist][s]["FULL"]            
#             ref_mean, ref_var = np.mean(ref_dist), np.var(ref_dist)
            
#             for W in ["MEAN", "UCB"]:
#                 for L in [0, 0.6, 0.8]:   
#                     emp_dist = chol_prune[nt][dist][s][(W, L)]
#                     emp_mean, emp_var = np.mean(emp_dist), np.var(emp_dist)
                   
#                     print("MC heuristic = {}".format(W + "-" + str(L)), file=dest)
#                     ks, p = ks_2samp(emp_dist, ref_dist)
#                     print("2-sided KS test: ({}, {})".format(ks, p), file=dest)
#                     mu_rel = abs(emp_mean - ref_mean)/ref_mean
#                     print("Relative error in mean (%): {}".format(mu_rel*100), file=dest)
#                     var_rel = abs(emp_var - ref_var)/ref_var
#                     print("Relative error in variance (%): {}".format(var_rel*100), file=dest)
#                     prune_time = chol_prune_timings[nt][dist][s][(W, L)][0]
#                     full_time = chol_prune_timings[nt][dist][s]["FULL"]
#                     time_reduction = 100 - (prune_time/full_time)*100
#                     print("Time savings (%): {}".format(time_reduction), file=dest)
#                     print("------------------------------------------------------", file=dest)

# Average summary.
s = 1000
with open("{}/chol_avgs_{}SAMPLES.txt".format(summary_path, s), "w") as dest:
    print("AVERAGE QUALITY OF PRUNED MC SOLUTIONS RELATIVE TO FULL MC SOLUTIONS.", file=dest)
    print("NUMBER OF SAMPLES: {}.".format(s), file=dest)
    
    for dist in ["NORMAL", "GAMMA", "UNIFORM"]:
        print("\n\n\n---------------------------------", file=dest)
        print("WEIGHT DISTRIBUTIONS: {}".format(dist), file=dest) 
        print("---------------------------------", file=dest)
        
        mean_errors = {WL : [] for WL in it.product(["MEAN", "UCB"], [0, 0.6, 0.8])}
        var_errors = {WL : [] for WL in it.product(["MEAN", "UCB"], [0, 0.6, 0.8])}
        ks_stats = {WL : [] for WL in it.product(["MEAN", "UCB"], [0, 0.6, 0.8])}
        time_reductions = {WL : [] for WL in it.product(["MEAN", "UCB"], [0, 0.6, 0.8])}
        
        for nt in n_tasks: 
             ref_dist = chol_prune[nt][dist][s]["FULL"] 
             ref_mean, ref_var = np.mean(ref_dist), np.var(ref_dist)             
             for W, L in it.product(["MEAN", "UCB"], [0, 0.6, 0.8]):
                emp_dist = chol_prune[nt][dist][s][(W, L)]
                emp_mean, emp_var = np.mean(emp_dist), np.var(emp_dist)
                mu_rel = abs(emp_mean - ref_mean)/ref_mean
                mean_errors[(W, L)].append(100*mu_rel)
                var_rel = abs(emp_var - ref_var)/ref_var
                var_errors[(W, L)].append(100*var_rel)
                ks, p = ks_2samp(emp_dist, ref_dist)
                ks_stats[(W, L)].append(ks)
                prune_time = chol_prune_timings[nt][dist][s][(W, L)][0]
                full_time = chol_prune_timings[nt][dist][s]["FULL"]
                time_reduction = 100 - (prune_time/full_time)*100
                time_reductions[(W, L)].append(time_reduction)
        
        # Compute averages over entire set.
        for WL in it.product(["MEAN", "UCB"], [0, 0.6, 0.8]):
            print("------------------------------------------------------", file=dest)
            print("MC heuristic = {}".format(WL), file=dest)
            print("Average relative error in mean (%): {}".format(np.mean(mean_errors[WL])), file=dest)
            print("Average relative error in variance (%): {}".format(np.mean(var_errors[WL])), file=dest)
            print("Average KS statistic: {}".format(np.mean(ks_stats[WL])), file=dest)
            print("Average time reduction (%): {}".format(np.mean(time_reductions[WL])), file=dest)
            print("------------------------------------------------------", file=dest)
    

                       
# Plots.
# # Relative error in mean.     
# colors = {"MEAN" : '#E24A33', "UCB" : '#348ABD'}         
# markers = {0 : 'o', 0.6 : 's', 0.8 : "D"}  
# styles = {0 : ':', 0.6 : '--', 0.8 : "-"}   
# labels = {("MEAN", 0) : 'M-CP', ("MEAN", 0.6) : 'M-60', ("MEAN", 0.8) : 'M-80', 
#           ("UCB", 0) : 'U-CP', ("UCB", 0.6) : 'U-60', ("UCB", 0.8) : 'U-80'} 
# add_plot_path = "../plots/pruning/chol_mean_error"
# pathlib.Path(add_plot_path).mkdir(parents=True, exist_ok=True)
# for dist, s in it.product(["NORMAL", "GAMMA", "UNIFORM"], [10, 30, 100, 1000]):
    
#     ref_means = list(np.mean(chol_prune[nt][dist][s]["FULL"]) for nt in n_tasks)
#     fig = plt.figure(dpi=400)
#     ax1 = fig.add_subplot(111)
#     for W, L in it.product(["MEAN", "UCB"], [0, 0.6, 0.8]):  
#         emp_means = list(np.mean(chol_prune[nt][dist][s][(W, L)]) for nt in n_tasks)
#         rel_errors = list(abs(e - r)/r for r, e in zip(ref_means, emp_means))
#         ax1.plot(n_tasks, rel_errors, color=colors[W], linestyle=styles[L], marker=markers[L], label=labels[(W, L)])
#     # plt.yscale('log')
#     ax1.set_xlabel("DAG SIZE", labelpad=5)
#     ax1.set_ylabel("REL. ERROR IN MEAN", labelpad=5)
#     ax1.legend(handlelength=2.5, handletextpad=0.3, ncol=2, loc='best', fancybox=True, facecolor='white') 
#     plt.savefig('{}/{}_{}SAMPLES'.format(add_plot_path, dist, s), bbox_inches='tight') 
#     plt.close(fig)                
                    
# # Relative error in variance.     
# colors = {"MEAN" : '#E24A33', "UCB" : '#348ABD'}         
# markers = {0 : 'o', 0.6 : 's', 0.8 : "D"}  
# styles = {0 : ':', 0.6 : '--', 0.8 : "-"}   
# labels = {("MEAN", 0) : 'M-CP', ("MEAN", 0.6) : 'M-60', ("MEAN", 0.8) : 'M-80', 
#           ("UCB", 0) : 'U-CP', ("UCB", 0.6) : 'U-60', ("UCB", 0.8) : 'U-80'} 
# add_plot_path = "../plots/pruning/chol_var_error"
# pathlib.Path(add_plot_path).mkdir(parents=True, exist_ok=True)
# for dist, s in it.product(["NORMAL", "GAMMA", "UNIFORM"], [10, 30, 100, 1000]):    
#     ref_vars = list(np.var(chol_prune[nt][dist][s]["FULL"]) for nt in n_tasks)
#     fig = plt.figure(dpi=400)
#     ax1 = fig.add_subplot(111)
#     for W, L in it.product(["MEAN", "UCB"], [0, 0.6, 0.8]):  
#         emp_vars = list(np.var(chol_prune[nt][dist][s][(W, L)]) for nt in n_tasks)
#         rel_errors = list(100*abs(e - r)/r for r, e in zip(ref_vars, emp_vars))
#         ax1.plot(n_tasks, rel_errors, color=colors[W], linestyle=styles[L], marker=markers[L], label=labels[(W, L)])
#     plt.yscale('log')
#     plt.xscale('log')
#     ax1.set_xlabel("DAG SIZE", labelpad=5)
#     ax1.set_ylabel("REL. ERROR IN VARIANCE (%)", labelpad=5)
#     ax1.legend(handlelength=2.5, handletextpad=0.3, ncol=2, loc='best', fancybox=True, facecolor='white') 
#     plt.savefig('{}/{}_{}SAMPLES'.format(add_plot_path, dist, s), bbox_inches='tight') 
#     plt.close(fig) 
                
# KS statistic.
# colors = {"MEAN" : '#E24A33', "UCB" : '#348ABD'}         
# markers = {0 : 'o', 0.6 : 's', 0.8 : "D"}  
# styles = {0 : ':', 0.6 : '--', 0.8 : "-"}   
# labels = {("MEAN", 0) : 'M-CP', ("MEAN", 0.6) : 'M-60', ("MEAN", 0.8) : 'M-80', 
#           ("UCB", 0) : 'U-CP', ("UCB", 0.6) : 'U-60', ("UCB", 0.8) : 'U-80'} 
# add_plot_path = "../plots/pruning/chol_ks_stat"
# pathlib.Path(add_plot_path).mkdir(parents=True, exist_ok=True)
# for dist, s in it.product(["NORMAL", "GAMMA", "UNIFORM"], [10, 30, 100, 1000]):    
#     ref_vars = list(np.var(chol_prune[nt][dist][s]["FULL"]) for nt in n_tasks)
#     fig = plt.figure(dpi=400)
#     ax1 = fig.add_subplot(111)
#     for W, L in it.product(["MEAN", "UCB"], [0, 0.6, 0.8]): 
#         ks_stats = []
#         for nt in n_tasks:   
#             ref_dist = chol_prune[nt][dist][s]["FULL"]
#             emp_dist = chol_prune[nt][dist][s][(W, L)]
#             ks, p = ks_2samp(emp_dist, ref_dist)
#             ks_stats.append(ks)
#         ax1.plot(n_tasks, ks_stats, color=colors[W], linestyle=styles[L], marker=markers[L], label=labels[(W, L)])
#     plt.xscale('log')
#     ax1.set_xlabel("DAG SIZE", labelpad=5)
#     ax1.set_ylabel("KS STATISTIC", labelpad=5)
#     ax1.legend(handlelength=2.5, handletextpad=0.3, ncol=2, loc='best', fancybox=True, facecolor='white') 
#     plt.savefig('{}/{}_{}SAMPLES'.format(add_plot_path, dist, s), bbox_inches='tight') 
#     plt.close(fig) 

# Histograms. 