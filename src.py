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
            # Need to be careful to make sure mu and var aren't zero (shouldn't be for a Gamma dist ofc but programmatically sometimes tempting.) 
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
            
        Phi_b = norm.cdf(b)
        Phi_minus = norm.cdf(-b)
        pdf_b = norm.pdf(b) 
        
        mu = self.mu * Phi_b + other.mu * Phi_minus + a * pdf_b      
        var = (self.mu**2 + self.var) * Phi_b
        var += (other.mu**2 + other.var) * Phi_minus
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
            
        Phi_b = norm.cdf(b)
        Phi_minus = norm.cdf(-b)
        pdf_b = norm.pdf(b) 
        
        mu = mu = self.mu * Phi_minus + other.mu * Phi_b - a * pdf_b     
        var = (self.mu**2 + self.var) * Phi_minus
        var += (other.mu**2 + other.var) * Phi_b
        var -= (self.mu + other.mu) * a * pdf_b
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
        if self.var is None:
            self.set_var()
        return "CRV(mu = {}, var = {})".format(self.mu, self.var)
    def __add__(self, other): 
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
    
    def set_var(self):
        self.var = sum(v**2 for v in self.alphas.values())
    
    def get_rho(self, other):
        other_alphas = defaultdict(float, other.alphas)
        dot_alphas = {}
        for k, v in self.alphas.items():
            dot_alphas[k] = self.alphas[k] * other_alphas[k] 
        n = sum(dot_alphas.values())   
        sd1 = np.sqrt(self.var)
        sd2 = np.sqrt(other.var)
        return n / (sd1 * sd2)
    
    def can_max(self, other):
        # Calculate rho.
        rho = self.get_rho(other)   
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
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """Graph is an NetworkX digraph with RV nodes and edge weights. Usually output by functions elsewhere..."""
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def realize(self, static=False, dist="NORMAL", percentile=None, fixed=set()):  
        """
        Realize all cost RVs between node with index first and node with index last (inclusive).
        Notes:
            1. Doesn't allow costs to be realized from different distribution types but may extend in future.
        TODO: better way to partially realize the DAG?
        """
                
        for t in self.top_sort:
            if t.ID not in fixed:               
                t.realize(static=static, dist=dist, percentile=percentile)
            for p in self.graph.predecessors(t):
                if (p.ID, t.ID) not in fixed:  
                    try:
                        self.graph[p][t]['weight'].realize(static=static, dist=dist, percentile=percentile) 
                    except AttributeError:  # Disjunctive edge weight is int/float (0/0.0). 
                        pass 
                
    def reset(self, fixed=set()):  
        """
        Realize all cost RVs between node with index first and node with index last (inclusive).
        Notes:
            1. Doesn't allow costs to be realized from different distribution types but may extend in future.
        """
                
        for t in self.top_sort:
            if t.ID not in fixed:
                t.realization = None
            for p in self.graph.predecessors(t):
                if (p.ID, t.ID) not in fixed:
                    try:
                        self.graph[p][t]['weight'].realization = None 
                    except AttributeError:  # Disjunctive edge weight is int/float (0/0.0). 
                        pass
                    
    def longest_path(self, pert_bound=False):
        """
        Computes the (realized) longest path of the DAG. If pert_bound == True, returns the classic PERT-CPM bound on the 
        expected value of the longest path instead.
        """            
        Z = {}       
        for t in self.top_sort:
            start_length = 0.0
            for p in self.graph.predecessors(t):
                m = Z[p.ID]
                try:
                    if pert_bound:
                        m += self.graph[p][t]['weight'].mu
                    else:
                        m += self.graph[p][t]['weight'].realization
                except AttributeError: # Disjunctive edge.
                    pass
                start_length = max(start_length, m)
            task_cost = t.mu if pert_bound else t.realization                
            Z[t.ID] = task_cost + start_length                                           
        return Z      

    def kamburowski(self):
        """TODO."""
        return
    
    def monte_carlo(self, samples=10, dist="NORMAL", fixed={}):
        """
        Monte Carlo sampling method to estimate the makespan distribution (well, its first two moments, since it is assumed to be 
        normal by the CLT) of the longest path. 
        """
        
        lps = []        
        for _ in range(samples):
            self.realize(dist=dist, fixed=fixed)
            Z = self.longest_path()
            lp = Z[self.top_sort[-1].ID]     # Assumes single exit task.
            lps.append(lp)
        mu = np.mean(lps)
        var = np.var(lps)
        self.reset(fixed=fixed)
        return RV(mu, var)

    def sculli(self, remaining=False):
        """
        Sculli's method for estimating the makespan of a fixed-cost stochastic DAG.
        'The completion time of PERT networks,'
        Sculli (1983).    
        """
        
        if remaining:
            R = {}
            backward_traversal = list(reversed(self.top_sort)) 
            for t in backward_traversal:
                children = list(self.graph.successors(t))
                try:
                    c = children[0]
                    m = self.graph[t][c]['weight'] + c + R[c.ID] 
                    for c in children[1:]:
                        m1 = self.graph[t][c]['weight'] + c + R[c.ID]
                        m = m.clark_max(m1, rho=0)
                    R[t.ID] = m  
                except IndexError:  # Entry task.
                    R[t.ID] = 0.0        
            return R            
        
        L = {}
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            try:
                p = parents[0]
                m = self.graph[p][t]['weight'] + L[p.ID] 
                for p in parents[1:]:
                    m1 = self.graph[p][t]['weight'] + L[p.ID]
                    m = m.clark_max(m1, rho=0)
                L[t.ID] = m + t 
            except IndexError:  # Entry task.
                L[t.ID] = t        
        return L   

    def corLCA(self, remaining=False, return_correlation_tree=False):
        """
        CorLCA heuristic for estimating the makespan of a fixed-cost stochastic DAG.
        'Correlation-aware heuristics for evaluating the distribution of the longest path length of a DAG with random weights,' 
        Canon and Jeannot (2016).     
        Assumes single entry and exit tasks.    
        """   
        
        if remaining:
            correlation_tree = nx.DiGraph()     
            R, C = {}, {} 
            backward_traversal = list(reversed(self.top_sort)) 
            for t in backward_traversal:
                dom_child = None
                for child in self.graph.successors(t):
                    R_ij = self.graph[t][child]['weight'] + R[child.ID] + child
                    C[(child.ID, t.ID)] = self.graph[t][child]['weight'] + C[child.ID] 
                    if dom_child is None:
                        dom_child = child
                        eta = R_ij
                    else:
                        get_lca = nx.algorithms.tree_all_pairs_lowest_common_ancestor(correlation_tree, pairs=[(dom_child.ID, child.ID)])
                        lca = list(get_lca)[0][1] 
                        r = C[lca].var / (np.sqrt(C[(dom_child.ID, t.ID)].var) * np.sqrt(C[(child.ID, t.ID)].var))
                        if R_ij.mu > eta.mu: 
                            dom_child = child
                        eta = eta.clark_max(R_ij, rho=r) 
                if dom_child is None:
                    R[t.ID] = 0.0
                    C[t.ID] = RV(t.mu, t.var)
                else:
                    R[t.ID] = eta
                    C[t.ID] = self.graph[t][dom_child]['weight'] + C[dom_child.ID] 
                    correlation_tree.add_edge(dom_child.ID, t.ID)
            if return_correlation_tree:
                return R, correlation_tree, C
            return R
                    
                    
        # Correlation tree.
        correlation_tree = nx.DiGraph()        
        # F represents finish times (called Y in 2016 paper). C is an approximation to F for estimating rho values. 
        L, C = {}, {} 
        
        # Traverse the DAG in topological order.        
        for t in self.top_sort:                
            
            dom_parent = None 
            for parent in self.graph.predecessors(t):                
                
                # L_ij = path length up to (but not including) node t.
                L_ij = self.graph[parent][t]['weight'] + L[parent.ID]         
                # Need to store C edge values to compute rho.
                C[(parent.ID, t.ID)] = self.graph[parent][t]['weight'] + C[parent.ID] 
                                    
                # Only one parent.
                if dom_parent is None:
                    dom_parent = parent 
                    eta = L_ij                    
                # At least two parents, so need to use Clark's equations to compute eta.
                else:                    
                    # Find the lowest common ancestor of the dominant parent and the current parent.
                    get_lca = nx.algorithms.tree_all_pairs_lowest_common_ancestor(correlation_tree, pairs=[(dom_parent.ID, parent.ID)])
                    lca = list(get_lca)[0][1]
                        
                    # Estimate the relevant correlation.
                    r = C[lca].var / (np.sqrt(C[(dom_parent.ID, t.ID)].var) * np.sqrt(C[(parent.ID, t.ID)].var))
                        
                    # Find dominant parent for the maximization.
                    # Assuming everything normal so it suffices to compare expected values.
                    if L_ij.mu > eta.mu: 
                        dom_parent = parent
                    
                    # Compute eta.
                    eta = eta.clark_max(L_ij, rho=r)  
            
            if dom_parent is None: # Entry task...
                L[t.ID] = RV(t.mu, t.var)
                C[t.ID] = RV(t.mu, t.var)         
            else:
                L[t.ID] = t + eta 
                C[t.ID] = t + self.graph[dom_parent][t]['weight'] + C[dom_parent.ID]
                # Add edge in correlation tree from the dominant parent to the current task.
                correlation_tree.add_edge(dom_parent.ID, t.ID) 
        if return_correlation_tree:
            return L, correlation_tree, C
        return L
    
    def corLCA_lite(self):
        """
        Faster but less accurate version of CorLCA.
        Bit of a misnomer in that the common ancestor found isn't necessarily the lowest.
        """    
        
        # Dominant ancestors dict used instead of DiGraph for the common ancestor queries.
        dominant_ancestors = defaultdict(list)        
        # L represents longest path estimates (called Y in 2016 paper). 
        L = {}
        
        # Traverse the DAG in topological order.        
        for t in self.top_sort:               
            
            dom_parent = None 
            for parent in self.graph.predecessors(t):
                
                # L_ij = path length up to (but not including) node t.
                L_ij = self.graph[parent][t]['weight'] + L[parent.ID]   
                                    
                # Only one parent.
                if dom_parent is None:
                    dom_parent = parent 
                    eta = L_ij
                    
                # At least two parents, so need to use Clark's equations to compute eta.
                else:                    
                    # Find the lowest common ancestor of the dominant parent and the current parent.
                    check_set = set(dominant_ancestors[parent.ID])
                    for a in reversed(dominant_ancestors[parent.ID]):
                        if a in check_set:
                            lca = a
                            break
                        
                    # Estimate the relevant correlation.
                    r = min(1, L[lca].var / (np.sqrt(eta.var) * np.sqrt(L_ij.var))) 
                        
                    # Find dominant parent for the maximization.
                    if L_ij.mu > eta.mu: 
                        dom_parent = parent
                    
                    # Compute eta.
                    eta = eta.clark_max(L_ij, rho=r)  
            
            if dom_parent is None: # Entry task...
                L[t.ID] = RV(t.mu, t.var)        
            else:
                L[t.ID] = t + eta 
                dominant_ancestors[t.ID] = dominant_ancestors[dom_parent.ID] + [dom_parent.ID] 
                
        return L #L[self.top_sort[-1].ID]     # Assumes single exit task.
    
    def canonical(self):
        """
        Estimates the makespan distribution using the canonical method...
        Assumes DAG has no edge weights and all costs are in canonical form.
        TODO: take another look at this.
        """
        
        # Convert to canonical form.
        CG = nx.DiGraph() 
        mapping = {} 
        i = 0          
        for t in self.top_sort:
            i += 1
            try:
                n = mapping[t.ID]
            except KeyError:
                n = CRV(t.mu, {i: np.sqrt(t.var)}, ID=i) # TODO - check all these IDs work.
                CG.add_node(n)
                mapping[t.ID] = n
                
            children = list(self.graph.successors(t))
            for c in children:
                i += 1
                m, s = self.graph[t][c]['weight'].mu, np.sqrt(self.graph[t][c]['weight'].var)
                e = CRV(m, {i : s}, ID=i) # TODO.
                CG.add_edge(n, e)
                try:
                    nc = mapping[c.ID]
                    CG.add_edge(e, nc)
                except KeyError:
                    i += 1
                    nc = CRV(c.mu, {i : np.sqrt(c.var)}, ID=i) # TODO.
                    CG.add_edge(e, nc)
                    mapping[c.ID] = nc
                    
        # Find longest path.
        L = {}
        new_top_sort = list(nx.topological_sort(CG)) 
        for t in new_top_sort:
            parents = list(self.graph.predecessors(t))
            try:
                p = parents[0]
                m = L[p.ID] 
                for p in parents[1:]:
                    m1 = L[p.ID]
                    m = m.can_max(m1)
                L[t.ID] = m + t 
            except IndexError:
                L[t.ID] = t
                    
        return L    
    
    def partially_realize(self, fraction, dist="NORMAL", percentile=None, return_info=False):
        """TODO."""
        # Realize entire DAG.
        self.realize(dist=dist, percentile=percentile)
        # Compute makespan.
        L = self.longest_path()
        # Find the "current" time.
        T = fraction * L[self.top_sort[-1].ID]
        # Determine which costs have been realized before time T.
        fixed = set()
        for t in self.top_sort:
            if L[t.ID] <= T:
                fixed.add(t.ID)
            else:
                t.realization = None
            for p in self.graph.predecessors(t):
                try:
                    if L[p.ID] + self.graph[p][t]['weight'].realization > T:
                        self.graph[p][t]['weight'].realization = None
                    else:
                        fixed.add((p.ID, t.ID))
                except AttributeError:
                    pass
        # DAG is now partially realized... 
        # Return finish times for tasks completed before T and fixed.
        if return_info:
            Z = {k : v for k, v in L.items() if v <= T} 
            return Z, fixed     
        
    def update_corLCA(self, L, correlation_tree, C):
        """
        L is dict {t.ID : N(mu, sigma)} of longest path/finish time estimates before runtime.
        correlation_tree is the correlation tree as defined by CorLCA.
        C is a dict {t.ID : N(mu1, sigma1)} of approximations to L which are used to estimate the correlations.
        """
        
        F = {}
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            if t.realization is not None:
                st = 0
                for p in parents:
                    try:
                        m = F[p.ID] + self.graph[p][t]['weight'].realization
                    except AttributeError:
                        m = F[p.ID]
                    st = max(st, m)
                F[t.ID] = t.realization + st
                continue
            # Task not realized, but parents may be...
            real_p, rv_p = {}, {}
            for p in parents:
                m = F[p.ID] + self.graph[p][t]['weight']
                try:
                    m.mu
                    rv_p[p.ID] = m
                except AttributeError:
                    real_p[p.ID] = m
            if len(real_p) == len(parents):
                F[t.ID] = t + max(real_p.values())
            elif len(rv_p) == len(parents):
                dom_parent = None
                for parent in self.graph.predecessors(t):   
                    F_ij = self.graph[parent][t]['weight'] + F[parent.ID]    
                    if dom_parent is None:
                        dom_parent = parent 
                        st = F_ij  
                    else:  # TODO: change correlation tree/C?
                        get_lca = nx.algorithms.tree_all_pairs_lowest_common_ancestor(correlation_tree, pairs=[(dom_parent.ID, parent.ID)])
                        lca = list(get_lca)[0][1]
                        r = C[lca].var / (np.sqrt(C[(dom_parent.ID, t.ID)].var) * np.sqrt(C[(parent.ID, t.ID)].var))
                        if F_ij.mu > st.mu: 
                            dom_parent = parent
                        st = st.clark_max(F_ij, rho=r) 
                F[t.ID] = t if dom_parent is None else t + st
                
            else: # TODO: why is b sometimes 0?
                X = max(real_p.values())
                # Find original maximization.
                M = L[t.ID] - t # TODO: should we recalculate L at this stage since some parents may have been updated?
                # Update M.
                a = (X - M.mu) / np.sqrt(M.var)
                print(a)
                pa = norm.pdf(a)
                b = 1 - norm.cdf(a) 
                print(b)
                mu_add = (np.sqrt(M.var) * pa) / b
                var_mult = 1 + (a * pa) / b - (pa/b)**2 
                Mdash = RV(M.mu + mu_add, M.var * var_mult)
                F[t.ID] = Mdash + t
                
        return F
                
                
                    