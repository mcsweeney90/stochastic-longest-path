#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions and classes for working with stochastic directed acyclic graphs (SDAGs).
"""

import dill
import networkx as nx
import numpy as np
from collections import defaultdict
from scipy.stats import norm

class RV:
    """
    Random variable class.
    Notes:
        - Defined by only mean and variance so can in theory be from any distribution but some functions
          e.g., addition and multiplication assume (either explicitly or implicitly) RV is Gaussian.
          (This is because addition/mult only done when RV represents a finish time estimate so is assumed to be
          roughly normal anyway.)
        - ID attribute isn't really necessary but occasionally makes things useful (e.g., for I/O).
    """
    def __init__(self, mu=0.0, var=0.0, realization=None, ID=None): 
        self.mu = mu
        self.var = var
        self.ID = ID
        self.realization = realization
    def __repr__(self):
        return "RV(mu = {}, var = {})".format(self.mu, self.var)
    def __add__(self, other): # Costs are typically independent so don't think there's any reason to ever consider correlations here.
        if isinstance(other, float) or isinstance(other, int):
            return RV(self.mu + other, self.var)
        return RV(self.mu + other.mu, self.var + other.var) 
    __radd__ = __add__ # Other way around...
    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return RV(self.mu - other, self.var)
        return RV(self.mu - other.mu, self.var + other.var)
    __rsub__ = __sub__ 
    def __mul__(self, c):
        return RV(c * self.mu, c * c * self.var)
    __rmul__ = __mul__ 
    def __truediv__(self, c): 
        return RV(self.mu / c, self.var / (c * c))
    __rtruediv__ = __truediv__ 
    def __floordiv__(self, c): 
        return RV(self.mu / c, self.var / (c * c))
    __rfloordiv__ = __floordiv__ 
    
    def reset(self):
        """Set all attributes except ID to their defaults."""
        self.mu, self.var, self.realization = 0, 0, None
        
    def realize(self, static=False, dist="NORMAL"):
        if static:
            self.realization = self.mu 
        elif dist == "NORMAL" or dist == "normal" or dist == "Normal" or dist == "Gaussian":             
            r = np.random.normal(self.mu, np.sqrt(self.var))
            if r < 0.0:
                r *= -1            
            self.realization = r
        elif dist == "GAMMA" or dist == "gamma" or dist == "Gamma":
            self.realization = np.random.gamma(shape=(self.mu**2 / self.var), scale=self.var/self.mu)
            # TODO: need to be careful to make sure mu and var aren't zero (shouldn't be for a Gamma dist ofc but programmatically sometimes tempting.) 
        elif dist == "uniform":
            self.realization = np.random.uniform(0, 2 * self.mu)
    
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
            
        Phi_b = norm.cdf(b)
        Phi_minus = norm.cdf(-b)
        Psi_b = norm.pdf(b) 
        
        mu = self.mu * Phi_b + other.mu * Phi_minus + a * Psi_b      
        var = (self.mu**2 + self.var) * Phi_b
        var += (other.mu**2 + other.var) * Phi_minus
        var += (self.mu + other.mu) * a * Psi_b
        var -= mu**2         
        return RV(mu, var)
    
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
            
        Phi_b = norm.cdf(b)
        Phi_minus = norm.cdf(-b)
        Psi_b = norm.pdf(b) 
        
        mu = mu = self.mu * Phi_minus + other.mu * Phi_b - a * Psi_b     
        var = (self.mu**2 + self.var) * Phi_minus
        var += (other.mu**2 + other.var) * Phi_b
        var -= (self.mu + other.mu) * a * Psi_b
        var -= mu**2         
        return RV(mu, var)  

class CRV:
    """
    Canonical representation of a random variable.
    TODO: takes too long, optimize. Add variance attribute?
    """
    def __init__(self, mu, alphas, ID=None):
        self.mu = mu
        self.alphas = alphas
        self.var = None
        self.ID = ID
    def __repr__(self):
        return "CRV(mu = {}, var = {})".format(self.mu, sum(s**2 for s in self.alphas.values()))
    def __add__(self, other): # Costs are typically independent so don't think there's any reason to ever consider correlations here.
        if isinstance(other, float) or isinstance(other, int):
            return CRV(self.mu + other, self.alphas)
        new_alphas = defaultdict(float)
        for k, v in self.alphas.items():
            new_alphas[k] += v
        for k, v in other.alphas.items():
            new_alphas[k] += v
        new_alphas = dict(new_alphas)
        new_crv = CRV(self.mu + other.mu, new_alphas) 
        new_crv.var = sum(v**2 for v in new_alphas.values())
        return new_crv
    __radd__ = __add__ # Other way around...
    
    def compute_var(self):
        self.var = sum(v**2 for v in self.alphas.values())
    
    def rho(self, other):
        other_alphas = defaultdict(float, other.alphas)
        dot_alphas = {}
        for k, v in self.alphas.items():
            dot_alphas[k] = self.alphas[k] * other_alphas[k] 
        n = sum(dot_alphas.values())   
        sd1 = np.sqrt(self.var)
        sd2 = np.sqrt(other.var)
        return n / (sd1 * sd2)
    
    def cmax(self, other):
        # Calculate rho.
        rho = self.rho(other)   
        # Calculate a.
        a = np.sqrt(self.var + other.var - 2 * np.sqrt(self.var) * np.sqrt(other.var) * rho)
        # Calculate b and the integrands.
        b = (self.mu - other.mu) / a         
        Phi_b = norm.cdf(b)
        Phi_minus = norm.cdf(-b)
        # Calculate new mu and alphas.
        new_mu = Phi_b * self.mu + Phi_minus * other.mu
        new_alphas = defaultdict(float)
        for k, v in self.alphas.items():
            new_alphas[k] += Phi_b * v
        for k, v in other.alphas.items():
            new_alphas[k] += Phi_minus * v
        new_alphas = dict(new_alphas) 
        new_crv = CRV(new_mu, new_alphas)
        new_crv.var = sum(v**2 for v in new_alphas.values())
        return new_crv
    
class SDAG:
    """Represents a stochastic graph."""
    def __init__(self, graph):
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.     
        
    def realize(self, start=0, finish=-1, static=False, dist="NORMAL"):  
        """
        Realize all costs.
        Notes:
            1. Doesn't allow costs to be realized from different distribution types but may extend in future.
        """
        
        if finish == -1: # TODO.
            finish = len(self.top_sort)
        
        for i, t in enumerate(self.top_sort):
            if i > finish:
                break
            if i < start:
                continue
            t.realize(static=static, dist=dist)
            for p in self.graph.predecessors(t):
                try:
                    self.graph[p][t]['weight'].realize(static=static, dist=dist) 
                except AttributeError:
                    pass
                    
    # def realize_remainder(self, dist=None):
    #     """
    #     Assuming that a subset of the tasks have been realized, realize the remainder and compute the 
    #     new makespan.
    #     TODO: only works with a specified distribution atm.
    #     """
    #     # New finish time function.
    #     real_F = {}
        
    #     # Find the "current" time. 
    #     time = 0.0        
    #     # Consider tasks in topological order.
    #     for task in self.top_sort:
    #         if task.realization is not None:
    #             real_F[task.ID] = task.realization 
    #             try:                        
    #                 st = max(real_F[p.ID] + self.graph[p][task]['weight'].realization for p in self.graph.predecessors(task))
    #                 real_F[task.ID] += st 
    #             except ValueError:
    #                 pass
    #             time = max(time, real_F[task.ID])
    #             continue
    #         # Realize task costs and incident edge costs until task finish time exceeds the current time.
    #         parents = list(self.graph.predecessors(task))
    #         st = 0.0
    #         # Realize the incident edge costs.
    #         for p in parents:
    #             edge_mu, edge_var = self.graph[p][task]['weight'].mu, self.graph[p][task]['weight'].var
    #             if edge_var == 0.0:
    #                 edge_realization = 0.0
    #             elif dist == "NORMAL" or dist == "normal" or dist == "Gaussian": 
    #                 edge_realization = np.random.normal(edge_mu, np.sqrt(edge_var)) # TODO: what if negative?
    #             elif dist == "GAMMA" or dist == "gamma":
    #                 edge_realization = np.random.gamma(shape=(edge_mu**2 / edge_var), scale=edge_var/edge_mu)                    
    #             st = max(st, real_F[p.ID] + edge_realization)
    #         # Realize the task cost.            
    #         if task.var == 0.0:
    #             task_realization = 0.0
    #         elif dist == "NORMAL" or dist == "normal" or dist == "Gaussian": 
    #             task_realization = np.random.normal(task.mu, np.sqrt(task.var)) # TODO: what if negative?
    #         elif dist == "GAMMA" or dist == "gamma":
    #             task_realization = np.random.gamma(shape=(task.mu**2 / task.var), scale=task.var/task.mu)
    #         # Compute the possible finish time.
    #         ft = st + task_realization
    #         if ft < time:
    #             ft = time + np.random.uniform(0, 1) * np.sqrt(task.var)                
    #         # Once an acceptable finish time is found, set real_F. 
    #         real_F[task.ID] = ft
                
    #     # Compute and return the makespan.
    #     return real_F[self.top_sort[-1].ID]     # Assumes single exit task.
                    
    def remove_edge_weights(self):
        """
        Convert the graph into an equivalent one without edge weights (but twice the number of vertices).
        Used in canonical method.
        TODO: Let n be #vertices of original graph and m the #edges. Then new graph has m + n vertices, is there
        an equivalent graph with fewer vertices?
        """
        
        G = nx.DiGraph() 
        mapping = {}
        entry_node = True
            
        for task in self.top_sort:
            if entry_node:
                n = RV(task.mu, task.var)
                G.add_node(n)
                mapping[task.ID] = n
                entry_node = False
            else:
                n = mapping[task.ID]
                
            children = list(self.graph.successors(task))
            for c in children:
                e = RV(self.graph[task][c]['weight'].mu, self.graph[task][c]['weight'].var) 
                G.add_edge(n, e)
                if c.ID not in mapping:
                    nc = RV(c.mu, c.var)
                    G.add_edge(e, nc)
                    mapping[c.ID] = nc
                else:
                    nc = mapping[c.ID]
                    G.add_edge(e, nc)
        
        # Now give all tasks a new ID.
        new_top_sort = list(nx.topological_sort(G))
        for i, task in enumerate(new_top_sort):
            task.ID = i

        self.graph = G
        self.top_sort = new_top_sort 
        
    def convert_costs_to_canonical_form(self):
        """
        Note ignores edge costs so implicitly assumes they're zero.
        Don't really need to do this explicitly but may occasionally be useful.
        """
        
        G = nx.DiGraph()
                    
        mapping = {}
        entry_node = True
            
        for task in self.top_sort:
            if entry_node:
                n = CRV(task.mu, {task.ID : np.sqrt(task.var)}, task.ID)
                n.var = task.var
                G.add_node(n)
                mapping[task.ID] = n
                entry_node = False
            else:
                n = mapping[task.ID]
            children = list(self.graph.successors(task))
            for c in children:
                if c.ID not in mapping:
                    nc = CRV(c.mu, {c.ID : np.sqrt(c.var)}, c.ID)
                    nc.var = c.var
                    G.add_edge(n, nc)
                    mapping[c.ID] = nc
                else:
                    nc = mapping[c.ID]
                    G.add_edge(n, nc)
        
        new_top_sort = list(nx.topological_sort(G))
        self.graph = G
        self.top_sort = new_top_sort    
                    
    def longest_path(self, expected=False):
        """
        Computes either the realized longest path of the DAG, assuming all costs have been realized, or
        if expected == False, computes an estimate of the makespan expected value by summing/maximizing 
        all expected values in the same way.
        """
            
        finish_times = {}       
        for task in self.top_sort:
            task_cost = task.mu if expected else task.realization
            edge_cost = 0.0
            for p in self.graph.predecessors(task):
                m = finish_times[p.ID]
                try:
                    if expected:
                        m += self.graph[p][task]['weight'].mu 
                    else:
                        m += self.graph[p][task]['weight'].realization
                except AttributeError:
                    pass
                edge_cost = max(edge_cost, m)
            finish_times[task.ID] = task_cost + edge_cost
                                           
        lp = finish_times[self.top_sort[-1].ID] # Assumes single exit task.    
        return lp 
    
    def monte_carlo(self, samples=10, dist="NORMAL"):
        """TODO"""
        
        lps = []        
        for _ in range(samples):
            self.realize(dist=dist)
            lp = self.longest_path()
            lps.append(lp)
        mu = np.mean(lps)
        var = np.var(lps)
        return RV(mu, var)

    def sculli(self):
        """
        Sculli's method for estimating the makespan of a fixed-cost stochastic DAG.
        'The completion time of PERT networks,'
        Sculli (1983).    
        """
        
        finish_times = {}
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            try:
                p = parents[0]
                m = self.graph[p][t]['weight'] + finish_times[p.ID] 
                for p in parents[1:]:
                    m1 = self.graph[p][t]['weight'] + finish_times[p.ID]
                    m = m.clark_max(m1, rho=0)
                finish_times[t.ID] = m + t 
            except IndexError:
                finish_times[t.ID] = t        
        return finish_times[self.top_sort[-1].ID]    

    def corLCA(self):
        """
        CorLCA heuristic for estimating the makespan of a fixed-cost stochastic DAG.
        'Correlation-aware heuristics for evaluating the distribution of the longest path length of a DAG with random weights,' 
        Canon and Jeannot (2016).     
        Assumes single entry and exit tasks.    
        Notes:
            - alternative method of computing LCA using dominant_ancestors dict isn't technically always the LCA, although it 
              does compute a "recent" common ancestor. However it is significantly faster.
        """   
                    
        # Correlation tree is a Networkx DiGraph, like self.graph.
        correlation_tree = nx.DiGraph()        
        # F represents finish times (called Y in 2016 paper). C is an approximation to F for estimating rho values. 
        F, C = {}, {} 
        
        # Traverse the DAG in topological order.        
        for task in self.top_sort:                
            
            dom_parent = None 
            for parent in self.graph.predecessors(task):
                
                # F(parent, task) = start time of task.
                F_ij = self.graph[parent][task]['weight'] + F[parent.ID]         
                # Need to store C edge values to compute rho.
                C[(parent.ID, task.ID)] = self.graph[parent][task]['weight'] + C[parent.ID] 
                                    
                # Only one parent.
                if dom_parent is None:
                    dom_parent = parent 
                    eta = F_ij
                    
                # At least two parents, so need to use Clark's equations to compute eta.
                else:                    
                    # Find the lowest common ancestor of the dominant parent and the current parent.
                    get_lca = nx.algorithms.tree_all_pairs_lowest_common_ancestor(correlation_tree, pairs=[(dom_parent.ID, parent.ID)])
                    lca = list(get_lca)[0][1]
                        
                    # Estimate the relevant correlation.
                    r = C[lca].var / (np.sqrt(C[(dom_parent.ID, task.ID)].var) * np.sqrt(C[(parent.ID, task.ID)].var))
                        
                    # Find dominant parent for the maximization.
                    # Assuming everything normal so it suffices to compare expected values.
                    if F_ij.mu > eta.mu: 
                        dom_parent = parent
                    
                    # Compute eta.
                    eta = eta.clark_max(F_ij, rho=r)  
            
            if dom_parent is None: # Entry task...
                F[task.ID] = RV(task.mu, task.var)
                C[task.ID] = RV(task.mu, task.var)         
            else:
                F[task.ID] = task + eta 
                C[task.ID] = task + self.graph[dom_parent][task]['weight'] + C[dom_parent.ID]
                # Add edge in correlation tree from the dominant parent to the current task.
                correlation_tree.add_edge(dom_parent.ID, task.ID)
                
        return F
    
    def lite_corLCA(self):
        """
        Faster but less accurate version of CorLCA.
        Bit of a misnomer in that the common ancestor found isn't necessarily the lowest.
        """    
        
        # Dominant ancestors dict used instead of DiGraph for the common ancestor queries.
        dominant_ancestors = defaultdict(list)        
        # F represents finish times (called Y in 2016 paper). 
        F = {}
        
        # Traverse the DAG in topological order.        
        for task in self.top_sort:               
            
            dom_parent = None 
            for parent in self.graph.predecessors(task):
                
                # F(parent, task) = start time of task.
                F_ij = self.graph[parent][task]['weight'] + F[parent.ID]   
                                    
                # Only one parent.
                if dom_parent is None:
                    dom_parent = parent 
                    eta = F_ij
                    
                # At least two parents, so need to use Clark's equations to compute eta.
                else:                    
                    # Find the lowest common ancestor of the dominant parent and the current parent.
                    check_set = set(dominant_ancestors[parent.ID])
                    for a in reversed(dominant_ancestors[parent.ID]):
                        if a in check_set:
                            lca = a
                            break
                        
                    # Estimate the relevant correlation.
                    r = min(1, F[lca].var / (np.sqrt(eta.var) * np.sqrt(F_ij.var))) 
                        
                    # Find dominant parent for the maximization.
                    if F_ij.mu > eta.mu: 
                        dom_parent = parent
                    
                    # Compute eta.
                    eta = eta.clark_max(F_ij, rho=r)  
            
            if dom_parent is None: # Entry task...
                F[task.ID] = RV(task.mu, task.var)        
            else:
                F[task.ID] = task + eta 
                dominant_ancestors[task.ID] = dominant_ancestors[dom_parent.ID] + [dom_parent.ID] 
                
        return F         
    
    def canonical(self):
        """
        Estimates the makespan distribution using the canonical method...
        Assumes DAG has no edge weights and all costs are in canonical form.
        """
            
        # F represents finish times. 
        finish_times = {}
        for task in self.top_sort:
            parents = list(self.graph.predecessors(task))
            try:
                p = parents[0]
                m = finish_times[p.ID] 
                for p in parents[1:]:
                    m1 = finish_times[p.ID]
                    m = m.cmax(m1)
                finish_times[task.ID] = m + task 
            except IndexError:
                finish_times[task.ID] = task
                    
        return finish_times         