#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create stochastic schedule DAGs for Cholesky DAGs.
TODO: tidy this all up and make clearer.
"""

import dill, pathlib, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
from timeit import default_timer as timer
from scipy.stats import norm
from Simulator import Platform

class RV:
    """
    Random variable class.
    Notes:
        - Defined by only mean and variance so can in theory be from any distribution but some functions
          e.g., addition and multiplication assume (either explicitly or implicitly) RV is Gaussian.
          (Addition/mult only done when RV represents a finish time/longest path estimate so is assumed to be
          at least roughly normal anyway.)
        - ID attribute isn't really necessary but occasionally makes things useful (e.g., for I/O).
    """
    def __init__(self, mu=0.0, var=0.0, realization=None, ID=None): 
        self.mu = mu
        self.var = var
        self.ID = ID
        self.realization = realization
    def __repr__(self):
        return "RV(mu = {}, var = {})".format(self.mu, self.var)
    # Overload addition operator.
    def __add__(self, other): 
        if isinstance(other, float) or isinstance(other, int):
            return RV(self.mu + other, self.var)
        return RV(self.mu + other.mu, self.var + other.var) 
    __radd__ = __add__ 
    # Overload subtraction operator.
    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return RV(self.mu - other, self.var)
        return RV(self.mu - other.mu, self.var + other.var)
    __rsub__ = __sub__ 
    # Overload multiplication operator.
    def __mul__(self, c):
        return RV(c * self.mu, c * c * self.var)
    __rmul__ = __mul__ 
    # Overload division operators.
    def __truediv__(self, c): 
        return RV(self.mu / c, self.var / (c * c))
    __rtruediv__ = __truediv__ 
    def __floordiv__(self, c): 
        return RV(self.mu / c, self.var / (c * c))
    __rfloordiv__ = __floordiv__ 
    
    def reset(self):
        """Set all attributes except ID to their defaults."""
        self.mu, self.var, self.realization = 0.0, 0.0, None
        
    def realize(self, static=False, dist="NORMAL", percentile=None):
        """
        Realize the RV from the specified distribution, according to the mean and expected value.
        If static, set realization to the mean.
        TODO: may add more distribution choices if they will be useful anywhere.
        Notes: percentile parameter assumes normal distribution. 
        """
        if static:
            self.realization = self.mu 
        elif dist == "NORMAL" or dist == "normal" or dist == "Normal" or dist == "Gaussian":    
            if percentile is None:
                r = np.random.normal(self.mu, np.sqrt(self.var))
                if r < 0.0:
                    r *= -1       
            else:
                r = norm.ppf(percentile, loc=self.mu, scale=np.sqrt(self.var))
            self.realization = r
        elif dist == "GAMMA" or dist == "gamma" or dist == "Gamma":
            self.realization = np.random.gamma(shape=(self.mu**2 / self.var), scale=self.var/self.mu) 
        elif dist == "uniform":
            u = np.sqrt(3 * self.var)
            r = np.random.uniform(-u, u)
            if r + self.mu < 0:
                r *= -1
            self.realization = self.mu + r
    
    def clark_max(self, other, rho=0):
        """
        Returns a new RV representing the maximization of self and other whose mean and variance
        are computed using Clark's equations for the first two moments of the maximization of two normal RVs.
        
        See:
        'The greatest of a finite set of random variables,'
        Charles E. Clark (1983).
        """
        a = np.sqrt(self.var + other.var - 2 * np.sqrt(self.var) * np.sqrt(other.var) * rho)     
        b = (self.mu - other.mu) / a
            
        cdf_b = norm.cdf(b)
        cdf_minus = norm.cdf(-b)
        pdf_b = norm.pdf(b) 
        
        mu = self.mu * cdf_b + other.mu * cdf_minus + a * pdf_b      
        var = (self.mu**2 + self.var) * cdf_b
        var += (other.mu**2 + other.var) * cdf_minus
        var += (self.mu + other.mu) * a * pdf_b
        var -= mu**2         
        return RV(mu, var)  # No ID set for new RV.
    
    def clark_min(self, other, rho=0):
        """
        Returns a new RV representing the minimization of self and other whose mean and variance
        are computed using Canon and Jeannot's extension of Clark's equations for the first two moments of the 
        maximization of two normal RVs.
        
        See:  
        'Precise evaluation of the efficiency and robustness of stochastic DAG schedules,'
        Canon and Jeannot (2009),        
        and
        'The greatest of a finite set of random variables,'
        Charles E. Clark (1963).
        """
        a = np.sqrt(self.var + other.var - 2 * np.sqrt(self.var) * np.sqrt(other.var) * rho)
        b = (self.mu - other.mu) / a
            
        cdf_b = norm.cdf(b)
        cdf_minus = norm.cdf(-b)
        pdf_b = norm.pdf(b) 
        
        mu = mu = self.mu * cdf_minus + other.mu * cdf_b - a * pdf_b     
        var = (self.mu**2 + self.var) * cdf_minus
        var += (other.mu**2 + other.var) * cdf_b
        var -= (self.mu + other.mu) * a * pdf_b
        var -= mu**2         
        return RV(mu, var) 

# Destinations etc.
platform = Platform(7, 1, name="Single_GPU")
chol_dag_path = 'task-graphs'

nbs = [128, 1024]
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]#, 16215, 22100]

for nb in nbs:
    print("\nStarting nb = {}".format(nb))
    chol_dag_dest = 'cholesky_heft_accelerated/nb{}'.format(nb)
    pathlib.Path(chol_dag_dest).mkdir(parents=True, exist_ok=True)
    # Load costs (for variances):
    with open('skylake_V100_samples/no_adt_nb{}.dill'.format(nb), 'rb') as file:
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
        with open('{}/nb{}/no_adt/{}tasks.dill'.format(chol_dag_path, nb, nt), 'rb') as file:
            dag = dill.load(file) 
            
        G = nx.DiGraph()
        mapping = {}
            
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
            n = RV(mu, var, ID=t.ID)
            G.add_node(n)
            mapping[t.ID] = n
        
        # Add the edges and their weights. 
        for t in dag.top_sort:
            n = mapping[t.ID]
            for s in list(dag.graph.successors(t)):
                c = mapping[s.ID]
                # Add edge.
                G.add_edge(n, c)
                # Add the costs.
                if t.where_scheduled == s.where_scheduled:
                    G[n][c]['weight'] = 0
                else:
                    source_type = platform.workers[t.where_scheduled].type
                    target_type = platform.workers[s.where_scheduled].type 
                    mu = t.comm_costs[source_type + target_type][s.ID]
                    var = variances[s.type][source_type + target_type]
                    w = RV(mu, var)
                    G[n][c]['weight'] = w
        
        # Add transitive edges.
        for p in platform.workers:
            if p.idle:
                continue
            elif len(p.load) == 1:
                continue
            for i, t in enumerate(p.load[:-1]):
                n = mapping[t[0]]
                s = p.load[i + 1][0]
                c = mapping[s]
                if not G.has_edge(n, c):
                    G.add_edge(n, c)
                    G[n][c]['weight'] = 0
                    
        # mkspan = dag.makespan()
        # print("\nDAG makespan: {}".format(mkspan))
                    
        # Reset DAG and platform if necessary.
        dag.reset()
        platform.reset() 
        
        # # Calculate longest path through G.
        # top_sort = list(nx.topological_sort(G)) 
        # finish_times = {}       
        # for t in top_sort:
        #     task_cost = t.mu 
        #     edge_cost = 0.0
        #     for p in G.predecessors(t):
        #         m = finish_times[p.ID]
        #         try:
        #             m += G[p][t]['weight'].mu
        #         except AttributeError:
        #             pass
        #         edge_cost = max(edge_cost, m)
        #     finish_times[t.ID] = task_cost + edge_cost
        # lp = finish_times[top_sort[-1].ID]
        # print("Longest path: {}".format(lp))
        
        # Save G.
        with open('{}/{}tasks.dill'.format(chol_dag_dest, nt), 'wb') as handle:
            dill.dump(G, handle)
                        
                    
                    
                
                
            

