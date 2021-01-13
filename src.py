#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions and classes for working with stochastic directed acyclic graphs (SDAGs).
"""

import dill, random
import networkx as nx
import numpy as np
from collections import defaultdict
from scipy.stats import norm, skew, kurtosis
from math import sqrt

# Uncomment if using timeout version of dodin_longest_paths.
# import time
# import timeout_decorator

class RV:
    """
    Random variable class.
    Notes:
        - Defined by only mean and variance so can in theory be from any distribution but some functions
          e.g., addition and multiplication assume (either explicitly or implicitly) RV is Gaussian.
          (Addition/mult only done when RV represents a finish time/longest path estimate so is assumed to be
          at least roughly normal anyway.)
        - ID attribute is often useful (e.g., for I/O).
        - Doesn't check that self.mu and self.var are nonzero for gamma. Obviously shouldn't be but sometimes tempting
          programmatically.
        - Random.random faster than numpy for individual realizations.
    """
    def __init__(self, mu=0.0, var=0.0, ID=None): 
        self.mu = mu
        self.var = var
        self.sd = sqrt(var)
        self.ID = ID
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
        self.mu, self.var, self.sd = 0.0, 0.0, 0.0    
    def realize(self, dist, static=False):
        if static:
            return self.mu 
        elif dist in ["N", "NORMAL", "normal"]:    
            r = random.gauss(self.mu, self.sd)    # Faster than numpy for individual realizations.         
            return r if r > 0.0 else -r
        elif dist in ["G", "GAMMA", "gamma"]:
            return random.gammavariate(alpha=(self.mu**2 / self.var), beta=self.var/self.mu)      
        elif dist in ["U", "UNIFORM", "uniform"]:
            u = sqrt(3) * self.sd
            r = self.mu + random.uniform(-u, u)                
            return r if r > 0.0 else -r 
    
class Path:
    """
    Path class - basically a collection of RVs.
    members is an (ordered) dict {task/edge ID : RV}
    """
    def __init__(self, length=RV(0.0, 0.0), members={}): 
        self.length = length
        self.members = members
        self.rep = None
    def __repr__(self): 
        return self.get_rep()  
    def __add__(self, other): # TODO: can creating new path be avoided?
        new = Path()
        new.length = self.length + other # Float, int or RV...
        new.members = self.members.copy()  
        try:
            new.members[other.ID] = RV(other.mu, other.var) # Biggest issue is that edges don't have IDs...
        except AttributeError:
            pass            
        return new
    __radd__ = __add__ 
    def get_rep(self):
        if self.rep is None:
            self.rep = ""
            for k in self.members.keys():
                if type(k) != int:
                    continue
                self.rep += (str(k) + "-") 
        return self.rep
    def get_rho(self, other):
        common_var = sum(self.members[w].var for w in self.members if w in other.members)             
        return common_var / (np.sqrt(self.length.var)*np.sqrt(other.length.var))    
    
class SDAG:
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """Graph is an NetworkX digraph with RV nodes and edge weights. Usually output by functions elsewhere..."""
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def set_random_weights(self, cov, dis_prob=0.1):
        """
        Used for setting weights for DAGs from the STG.
        """
        
        mubar = np.random.uniform(1, 100)        
        for t in self.top_sort:
            mu = np.random.gamma(shape=1.0, scale=mubar)
            eps = np.random.uniform(0.9, 1.1)
            sig = eps * cov * mu
            var = sig**2
            self.graph.nodes[t]['weight'] = RV(mu, var)
            for p in self.graph.predecessors(t):
                r = np.random.uniform(0, 1)
                if r < dis_prob:
                    self.graph[p][t]['weight'] = 0.0
                else:
                    mu = np.random.gamma(shape=1.0, scale=mubar)
                    eps = np.random.uniform(0.9, 1.1)
                    sig = eps * cov * mu
                    var = sig**2
                    self.graph[p][t]['weight'] = RV(mu, var)
    
    def monte_carlo(self, samples, dist="NORMAL", return_paths=False):
        """
        Monte Carlo method to estimate the distribution of the longest path. 
        TODO: return paths.
        """   
        
        E = []
        for _ in range(samples):
            L = {}
            for t in self.top_sort:
                w = self.graph.nodes[t]['weight'].realize(dist=dist) 
                st = 0.0
                parents = list(self.graph.predecessors(t))
                for p in parents:
                    pst = L[p]
                    try:
                        pst += self.graph[p][t]['weight'].realize(dist=dist)
                    except AttributeError: # Disjunctive edge.
                        pass
                    st = max(st, pst)
                L[t] = st + w
            E.append(L[self.top_sort[-1]])  # Assumes single exit task.
        return E
    
    def np_mc(self, samples, dist="NORMAL"):
        """
        Numpy version of MC.
        TODO: - Memory limit assumes 16GB RAM, check Matt's machine.
        TODO: no check if positive!
        """
                
        x = self.size * samples
        mem_limit = 1800000000
        if x < mem_limit:        
            L = {}
            for t in self.top_sort:
                m, s = self.graph.nodes[t]['weight'].mu, self.graph.nodes[t]['weight'].sd
                if dist in ["N", "NORMAL", "normal"]:  
                    w = np.random.normal(m, s, samples)
                elif dist in ["G", "GAMMA", "gamma"]:
                    v = self.graph.nodes[t]['weight'].var
                    sh, sc = (m * m)/v, v/m
                    w = np.random.gamma(sh, sc, samples)
                elif dist in ["U", "UNIFORM", "uniform"]:
                    u = sqrt(3) * s
                    w = np.random.uniform(-u + m, u + m, samples) 
                parents = list(self.graph.predecessors(t))
                if not parents:
                    L[t] = w 
                    continue
                pmatrix = []
                for p in self.graph.predecessors(t):
                    try:
                        m, s = self.graph[p][t]['weight'].mu, self.graph[p][t]['weight'].sd
                        if dist in ["N", "NORMAL", "normal"]: 
                            e = np.random.normal(m, s, samples)
                        elif dist in ["G", "GAMMA", "gamma"]:
                            v = self.graph[p][t]['weight'].var
                            sh, sc = (m * m)/v, v/m
                            e = np.random.gamma(sh, sc, samples)
                        elif dist in ["U", "UNIFORM", "uniform"]:
                            u = sqrt(3) * s
                            e = np.random.uniform(-u + m, u + m, samples)  
                        pmatrix.append(np.add(L[p], e))
                    except AttributeError:
                        pmatrix.append(L[p])
                st = np.amax(pmatrix, axis=0)
                L[t] = np.add(w, st)
            return L[self.top_sort[-1]] 
        else:
            E = []
            mx_samples = mem_limit//self.size
            runs = samples//mx_samples
            # print(mx_samples, runs)
            extra = samples % mx_samples
            for _ in range(runs):
                E += list(self.np_mc(samples=mx_samples, dist=dist))
            E += list(self.np_mc(samples=extra, dist=dist))
            return E 
    
    def CPM(self, variance=False):
        """
        Returns the classic PERT-CPM bound on the expected value of the longest path.
        If variance == True, also returns the variance of this path to use as a rough estimate
        of the longest path variance.
        """
        C = {}       
        for t in self.top_sort:
            st = 0.0
            if variance:
                v = 0.0
            for p in self.graph.predecessors(t):
                pst = C[p] if not variance else C[p].mu
                try:
                    pst += self.graph[p][t]['weight'].mu
                except AttributeError: # Disjunctive edge.
                    pass
                st = max(st, pst)  
                if variance and st == pst:
                    v = C[p].var
                    try:
                        v += self.graph[p][t]['weight'].var
                    except AttributeError:
                        pass
            m = self.graph.nodes[t]['weight'].mu
            if not variance:
                C[t] = m + st      
            else:
                var = self.graph.nodes[t]['weight'].var
                C[t] = RV(m + st, var + v)                                    
        return C    
    
    def kamburowski(self):
        """
        Returns:
            - lm, lower bounds on the mean. Dict in the form {task ID : m_underline}.
            - um, upper bounds on the mean. Dict in the form {task ID : m_overline}.
            - ls, lower bounds on the variance. Dict in the form {task ID : s_underline}.
            - us, upper bounds on the variance. Dict in the form {task ID : s_overline}.
        """
        lm, um, ls, us = {},{}, {}, {}
        for t in self.top_sort:
            nw = self.graph.nodes[t]['weight']
            parents = list(self.graph.predecessors(t))
            # Entry task(s).
            if not parents:
                lm[t], um[t] = nw.mu, nw.mu
                ls[t], us[t] = nw.var, nw.var
                continue
            # Lower bound on variance.
            if len(parents) == 1:
                ls[t] = ls[parents[0]] + nw.var
                try:
                    ls[t] += self.graph[parents[0]][t]['weight'].var
                except AttributeError:
                    pass
            else:
                ls[t] = 0.0
            # Upper bound on variance.
            v = 0.0
            for p in parents:
                sv = us[p] + nw.var
                try:
                    sv += self.graph[p][t]['weight'].var
                except AttributeError:
                    pass
                v = max(v, sv)
            us[t] = v
            # Lower bound on mean.
            Xunder = []
            for p in parents:
                pmu = lm[p] + nw.mu
                pvar = ls[p] + nw.var
                try:
                    pmu += self.graph[p][t]['weight'].mu
                    pvar += self.graph[p][t]['weight'].var
                except AttributeError:
                    pass
                Xunder.append(RV(pmu, pvar))
            Xunder = list(sorted(Xunder, key=lambda x:x.var))
            lm[t] = funder(Xunder)
            # Upper bound on mean.
            Xover = []
            for p in parents:
                pmu = um[p] + nw.mu
                pvar = us[p] + nw.var
                try:
                    pmu += self.graph[p][t]['weight'].mu
                    pvar += self.graph[p][t]['weight'].var
                except AttributeError:
                    pass
                Xover.append(RV(pmu, pvar))
            Xover = list(sorted(Xover, key=lambda x:x.var))
            um[t] = fover(Xover)
        
        return lm, um, ls, us      

    def sculli(self, direction="downward"):
        """
        Sculli's method for estimating the makespan of a fixed-cost stochastic DAG.
        'The completion time of PERT networks,'
        Sculli (1983).    
        """
        
        L = {}
        if direction == "downward":
            for t in self.top_sort:
                parents = list(self.graph.predecessors(t))
                try:
                    p = parents[0]
                    m = self.graph[p][t]['weight'] + L[p] 
                    for p in parents[1:]:
                        m1 = self.graph[p][t]['weight'] + L[p]
                        m = clark(m, m1, rho=0)
                    L[t] = m + self.graph.nodes[t]['weight']
                except IndexError:  # Entry task.
                    L[t] = self.graph.nodes[t]['weight']  
        elif direction == "upward":
            backward_traversal = list(reversed(self.top_sort)) 
            for t in backward_traversal:
                children = list(self.graph.successors(t))
                try:
                    s = children[0]
                    m = self.graph[t][s]['weight'] + self.graph.nodes[s]['weight'] + L[s] 
                    for s in children[1:]:
                        m1 = self.graph[t][s]['weight'] + self.graph.nodes[s]['weight'] + L[s]
                        m = clark(m, m1, rho=0)
                    L[t] = m  
                except IndexError:  # Entry task.
                    L[t] = 0.0   
        return L
            
    
    def corLCA(self, direction="downward", return_correlation_info=False):
        """
        CorLCA heuristic for estimating the makespan of a fixed-cost stochastic DAG.
        'Correlation-aware heuristics for evaluating the distribution of the longest path length of a DAG with random weights,' 
        Canon and Jeannot (2016).     
        Assumes single entry and exit tasks. 
        This is a fast version that doesn't explicitly construct the correlation tree.
        TODO: make sure upward version works.
        """    
        
        # Dominant ancestors dict used instead of DiGraph for the common ancestor queries. 
        # L represents longest path estimates. V[task ID] = variance of longest path of dominant ancestors (used to estimate rho).
        dominant_ancestors, L, V = {}, {}, {}
        
        if direction == "downward":      
            for t in self.top_sort:     # Traverse the DAG in topological order. 
                nw = self.graph.nodes[t]['weight']
                dom_parent = None 
                for parent in self.graph.predecessors(t):
                    pst = self.graph[parent][t]['weight'] + L[parent]   
                                        
                    # First parent.
                    if dom_parent is None:
                        dom_parent = parent 
                        dom_parent_ancs = set(dominant_ancestors[dom_parent])
                        dom_parent_sd = V[dom_parent]
                        try:
                            dom_parent_sd += self.graph[dom_parent][t]['weight'].var
                        except AttributeError:
                            pass
                        dom_parent_sd = sqrt(dom_parent_sd) 
                        st = pst
                        
                    # At least two parents, so need to use Clark's equations to compute eta.
                    else:                    
                        # Find the lowest common ancestor of the dominant parent and the current parent.
                        for a in reversed(dominant_ancestors[parent]):
                            if a in dom_parent_ancs:
                                lca = a
                                break
                            
                        # Estimate the relevant correlation.
                        parent_sd = V[parent]
                        try:
                            parent_sd += self.graph[parent][t]['weight'].var
                        except AttributeError:
                            pass
                        parent_sd = sqrt(parent_sd) 
                        r = V[lca] / (dom_parent_sd * parent_sd)
                            
                        # Find dominant parent for the maximization.
                        if pst.mu > st.mu: 
                            dom_parent = parent
                            dom_parent_ancs = set(dominant_ancestors[parent])
                            dom_parent_sd = parent_sd
                        
                        # Compute eta.
                        st = clark(st, pst, rho=r)  
                
                if dom_parent is None: # Entry task...
                    L[t] = nw  
                    V[t] = nw.var
                    dominant_ancestors[t] = [t]
                else:
                    L[t] = nw + st 
                    V[t] = dom_parent_sd**2 + nw.var
                    dominant_ancestors[t] = dominant_ancestors[dom_parent] + [t] 
                    
        elif direction == "upward":
            backward_traversal = list(reversed(self.top_sort)) 
            for t in backward_traversal:    
                dom_child = None 
                for child in self.graph.successors(t):
                    cw = self.graph.nodes[child]['weight']
                    cst = self.graph[t][child]['weight'] + cw + L[child]  
                    if dom_child is None:
                        dom_child = child 
                        dom_child_descs = set(dominant_ancestors[dom_child])
                        dom_child_sd = V[dom_child] + self.graph.nodes[dom_child]['weight'].var
                        try:
                            dom_child_sd += self.graph[t][dom_child]['weight'].var
                        except AttributeError:
                            pass
                        dom_child_sd = sqrt(dom_child_sd) 
                        st = cst
                    else: 
                        for a in reversed(dominant_ancestors[child]):
                            if a in dom_child_descs:
                                lca = a
                                break
                        child_sd = V[child] + self.graph.nodes[child]['weight'].var 
                        try:
                            child_sd += self.graph[t][child]['weight'].var
                        except AttributeError:
                            pass
                        child_sd = sqrt(child_sd) 
                        # Find LCA task. TODO: don't like this, rewrite so not necessary.
                        for s in self.top_sort:
                            if s == lca:
                                lca_var = self.graph.nodes[s]['weight'].var  
                                break
                        r = (V[lca] + lca_var) / (dom_child_sd * child_sd) 
                        if cst.mu > st.mu: 
                            dom_child = child
                            dom_child_descs = set(dominant_ancestors[child])
                            dom_child_sd = child_sd
                        st = clark(st, cst, rho=r)  
                if dom_child is None: # Entry task...
                    L[t], V[t] = 0.0, 0.0
                    dominant_ancestors[t] = [t]
                else:
                    L[t] = st 
                    V[t] = dom_child_sd**2 
                    dominant_ancestors[t] = dominant_ancestors[dom_child] + [t]            
        
        if return_correlation_info:
            return L, dominant_ancestors, V
        return L     
    
    def bootstrap_confidence_intervals(self, samples, resamples, dist="NORMAL"):
        """TODO."""
        # Get the initial sample and compute its mean.
        lps = self.monte_carlo(samples=samples, dist=dist)
        mu = np.mean(lps)
        # Re-sample.
        deltas = []
        for _ in range(resamples):
            R = np.random.choice(lps, size=samples, replace=True)
            star = np.mean(R)
            deltas.append(star - mu)
        # Sort deltas.
        deltas = list(sorted(deltas))
        intervals = {}
        # 80%.
        s = resamples // 10
        p = deltas[s - 1]
        q = deltas[9*s - 1]
        intervals[80] = (mu - q, mu - p)
        # 90%.
        s = resamples // 20
        p = deltas[s - 1]
        q = deltas[9*s - 1]
        intervals[90] = (mu - q, mu - p)
        # 95%.
        s = resamples // 40
        p = deltas[s - 1]
        q = deltas[9*s - 1]
        intervals[95] = (mu - q, mu - p)        
        return intervals
    
    def number_of_paths(self):
        """
        Count the number of paths through DAG.
        (Typically only used to show how enormous and impractical it is.)
        """        
        paths = {}
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            if not parents:
                paths[t] = 1
            else:
                paths[t] = sum(paths[p] for p in parents)                
        return paths                 
    
    # @timeout_decorator.timeout(5, timeout_exception=StopIteration)    # Uncomment if using timeout version (and imports at top).
    def dodin_critical_paths(self, epsilon=0.1, K=None, correlations=True):
        """
        TODO: update to reflect changes.
        Implementation is really poor atm since just proof of concept but will likely always be expensive...
        Is this really Dodin's algorithm in disguise?
        """
        
        # Compute comparison for determining if path is retained.
        x = norm.ppf(epsilon)
        
        candidates = {}        
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            # If entry node, create path.
            if not parents:
                candidates[t.ID] = [Path() + t]
            else: 
                # Identify path with greatest expected value.
                paths_by_parent = {}
                max_path, max_parent = Path(), None
                for p in parents:
                    paths_by_parent[p.ID] = []
                    edge_weight = self.graph[p][t]['weight'] 
                    for pt in candidates[p.ID]:
                        pth = pt + edge_weight + t 
                        if pth.length.mu > max_path.length.mu:
                            max_path = pth
                            max_parent = p.ID
                        paths_by_parent[p.ID].append(pth) 
                # Retain only non-dominated paths.
                candidates[t.ID] = []
                if K is not None:
                    probs = {}
                for p in parents:                        
                    for pth in paths_by_parent[p.ID]:                         
                        if p.ID == max_parent:
                            candidates[t.ID].append(pth)
                            if K is not None:
                                if pth == max_path:
                                    probs[pth] = float("inf")
                                else:
                                    r = max_path.get_rho(pth) if correlations else 0
                                    num = pth.length.mu - max_path.length.mu
                                    denom = np.sqrt(max_path.length.var + pth.length.var - r * np.sqrt(max_path.length.var)*np.sqrt(pth.length.var))
                                    y = num/denom
                                    probs[pth] = y
                        else:
                            r = max_path.get_rho(pth) if correlations else 0
                            num = pth.length.mu - max_path.length.mu
                            denom = np.sqrt(max_path.length.var + pth.length.var - r * np.sqrt(max_path.length.var)*np.sqrt(pth.length.var))
                            y = num/denom
                            if y > x:
                                candidates[t.ID].append(pth)
                                if K is not None:
                                    probs[pth] = y
                # If #candidates > limit, sort and retain only the greatest .
                if K is not None and len(candidates[t.ID]) > K:
                    candidates[t.ID] = list(reversed(sorted(candidates[t.ID], key=lambda pth:probs[pth])))
                    candidates[t.ID] = candidates[t.ID][:K]                    
        # Return set of path candidates terminating at (single) exit task.        
        return candidates[self.top_sort[-1].ID] 
    
    def static_critical_paths(self, K, weights="MEAN"):
        """
        TODO: update to reflect changes.
        Get the longest paths according to some average of the weights.
        TODO: really poor implementation. 
        """
        
        candidates = {}        
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            # Identify all possible paths.
            if not parents:
                candidates[t.ID] = [Path() + t]
            else: 
                # Get all possible paths.
                candidates[t.ID] = []
                for p in parents:
                    edge_weight = self.graph[p][t]['weight'] 
                    # try:
                    #     edge_weight.ID = (p.ID, t.ID)
                    # except AttributeError:
                    #     pass
                    for pt in candidates[p.ID]:
                        pth = pt + edge_weight + t 
                        candidates[t.ID].append(pth) 
                # Sort paths according to average.    
                if weights == "MEAN":
                    candidates[t.ID] = list(reversed(sorted(candidates[t.ID], key=lambda pth:pth.length.mu)))
                elif weights == "UCB":
                    candidates[t.ID] = list(reversed(sorted(candidates[t.ID], key=lambda pth:pth.length.mu + np.sqrt(pth.length.var))))
                if len(parents) > 1:                    
                    candidates[t.ID] = candidates[t.ID][:K]                                     
        # Return set of path candidates terminating at (single) exit task.        
        return candidates[self.top_sort[-1].ID] 
    
    def get_static_node_criticalities(self, weights="MEAN"):
        """
        TODO: update to reflect changes.
        Parameters
        ----------
        weights : TYPE, optional
            DESCRIPTION. The default is "mean".
        method : TYPE, optional
            DESCRIPTION. The default is "static".

        Returns
        -------
        None.

        """
        
        # Compute upward rank of all tasks.
        upward = {}
        backward_traversal = list(reversed(self.top_sort))  
        for t in backward_traversal:
            if weights == "mean" or weights == "MEAN":
                upward[t.ID] = t.mu
            elif weights == "UCB":
                upward[t.ID] = t.mu + np.sqrt(t.var)                
            children = list(self.graph.successors(t))
            mx = 0.0
            for c in children:
                try:
                    if weights == "mean" or weights == "MEAN":
                        edge_weight = self.graph[t][c]['weight'].mu 
                    elif weights == "UCB":
                        edge_weight = self.graph[t][c]['weight'].mu + np.sqrt(self.graph[t][c]['weight'].var)
                except AttributeError:
                    edge_weight = 0.0
                mx = max(mx, edge_weight + upward[c.ID])
            upward[t.ID] += mx
            
        # Compute downward rank of all tasks.
        downward, criticalities = {}, {}
        for t in self.top_sort:
            downward[t.ID] = 0.0
            parents = list(self.graph.predecessors(t))
            mx = 0.0
            for p in parents:
                pw = p.mu + np.sqrt(p.var) if weights == "UCB" else p.mu
                try:
                    if weights == "mean" or weights == "MEAN":
                        edge_weight = self.graph[p][t]['weight'].mu 
                    elif weights == "UCB":
                        edge_weight = self.graph[p][t]['weight'].mu + np.sqrt(self.graph[p][t]['weight'].var)
                except AttributeError:
                    edge_weight = 0.0
                mx = max(mx, pw + edge_weight + downward[p.ID])
            downward[t.ID] += mx
            # Calculate criticality.
            c = upward[t.ID] + downward[t.ID]
            criticalities[t.ID] = c
            
        return criticalities
    
    def get_critical_subgraph(self, f=0.9, node_limit=None, weights="MEAN"):
        """
        TODO: update to reflect changes.
        f controls the number of nodes to retain.
        """
        
        # Get node criticalities.
        criticalities = self.get_static_node_criticalities(weights=weights)
         
        # Identify nodes to be retained. 
        x = criticalities[self.top_sort[0].ID] # Assumes single entry node.
        if node_limit is not None:  # TODO: ignore f?
            cp_nodes, other_nodes = [], []
            for t in range(self.size):
                if abs(criticalities[t] - x) < 1e-6:
                    cp_nodes.append(t)      # Retain all critical path nodes at a minimum.
                else:
                    other_nodes.append(t)
            L = int(node_limit * self.size)
            y = L - len(cp_nodes)
            if y <= 0:
                retain = set(cp_nodes)
            else:
                node_sort = list(reversed(sorted(other_nodes, key=lambda n:criticalities[n])))
                retain = set(cp_nodes + node_sort[:y])
        else:
            y = f * x
            retain = set(t for t in range(self.size) if criticalities[t] > y)
                        
        # Construct subgraph.
        N, mapping = nx.DiGraph(), {}
        for t in self.top_sort:
            if t.ID not in retain:
                continue
            n = RV(t.mu, t.var, ID=t.ID)
            N.add_node(n)
            mapping[t.ID] = n
            for p in self.graph.predecessors(t):
                if p.ID not in retain:
                    continue
                q = mapping[p.ID]
                N.add_edge(q, n)
                try:
                    self.graph[p][t]['weight'].mu
                    edge_weight = RV(self.graph[p][t]['weight'].mu, self.graph[p][t]['weight'].var, ID=(p.ID, t.ID))
                except AttributeError:
                    edge_weight = 0.0
                N[q][n]['weight'] = edge_weight
        
        # Check that graph is fully connected.
        # TODO: just delete the nodes?
        source = mapping[self.top_sort[0].ID]
        sink = mapping[self.top_sort[-1].ID]
        for t in self.top_sort:
            if t.ID not in retain:
                continue
            if t.ID in [self.top_sort[0].ID, self.top_sort[-1].ID]:
                continue
            n = mapping[t.ID]
            if not len(list(N.predecessors(n))):
                N.add_edge(source, n)
                N[source][n]['weight'] = 0.0
            if not len(list(N.successors(n))):
                N.add_edge(n, sink)
                N[n][sink]['weight'] = 0.0            
                
        # Convert to SDAG object and return. 
        S = SDAG(N)
        return S       
    
    def partially_realize(self, fraction, dist="NORMAL", percentile=None, return_info=False):
        """
        TODO: update to reflect changes.
        """
        # Realize entire DAG.
        self.realize(dist=dist, percentile=percentile)
        # Compute makespan.
        L = self.real_longest_path()
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
        
    # def update_corLCA(self, L, correlation_tree, C):
    #     """
    #     L is dict {t.ID : N(mu, sigma)} of longest path/finish time estimates before runtime.
    #     correlation_tree is the correlation tree as defined by CorLCA.
    #     C is a dict {t.ID : N(mu1, sigma1)} of approximations to L which are used to estimate the correlations.
    #     TODO: still working on this.
    #     """
        
    #     F = {}
    #     for t in self.top_sort:
    #         parents = list(self.graph.predecessors(t))
    #         if t.realization is not None:
    #             st = 0
    #             for p in parents:
    #                 try:
    #                     m = F[p.ID] + self.graph[p][t]['weight'].realization
    #                 except AttributeError:
    #                     m = F[p.ID]
    #                 st = max(st, m)
    #             F[t.ID] = t.realization + st
    #             continue
    #         # Task not realized, but parents may be...
    #         real_p, rv_p = {}, {}
    #         for p in parents:
    #             m = F[p.ID] + self.graph[p][t]['weight']
    #             try:
    #                 m.mu
    #                 rv_p[p.ID] = m
    #             except AttributeError:
    #                 real_p[p.ID] = m
    #         if len(real_p) == len(parents): # All parents realized.
    #             F[t.ID] = t + max(real_p.values()) # TODO: else compute maximum no matter what and then do truncated Gaussian?
    #         elif len(rv_p) == len(parents): # No parents realized. TODO.
    #             dom_parent = None
    #             for parent in self.graph.predecessors(t):   
    #                 F_ij = self.graph[parent][t]['weight'] + F[parent.ID]    
    #                 if dom_parent is None:
    #                     dom_parent = parent 
    #                     st = F_ij  
    #                 else:  # TODO: change correlation tree/C?
    #                     get_lca = nx.algorithms.tree_all_pairs_lowest_common_ancestor(correlation_tree, pairs=[(dom_parent.ID, parent.ID)])
    #                     lca = list(get_lca)[0][1]
    #                     r = C[lca].var / (np.sqrt(C[(dom_parent.ID, t.ID)].var) * np.sqrt(C[(parent.ID, t.ID)].var))
    #                     if F_ij.mu > st.mu: 
    #                         dom_parent = parent
    #                     st = st.clark_max(F_ij, rho=r) 
    #             F[t.ID] = t if dom_parent is None else t + st
                
    #         else: 
    #             X = max(real_p.values())
    #             # Find original maximization.
    #             M_mu = L[t.ID].mu - t.mu
    #             M_var = L[t.ID].var - t.var
    #             # Update M.
    #             a = (X - M_mu) / np.sqrt(M_var)
    #             # print("\n", a)
    #             pa = norm.pdf(a)
    #             b = 1 - norm.cdf(a) 
    #             # print(b)
    #             mu_add = (np.sqrt(M_var) * pa) / b
    #             var_mult = 1 + (a * pa) / b - (pa/b)**2 
    #             Mdash = RV(M_mu + mu_add, M_var * var_mult)
    #             # print(mu_add, var_mult)
    #             F[t.ID] = Mdash + t               
                
    #     return F
    
# =============================================================================
# Functions.
# =============================================================================

def clark(r1, r2, rho=0, minimization=False):
    """
    Returns a new RV representing the maximization of self and other whose mean and variance
    are computed using Clark's equations for the first two moments of the maximization of two normal RVs.
    
    See:
    'The greatest of a finite set of random variables,'
    Charles E. Clark (1983).
    """
    a = sqrt(r1.var + r2.var - 2 * r1.sd * r2.sd * rho)     
    b = (r1.mu - r2.mu) / a            
    cdf = norm.cdf(b)
    mcdf = 1 - cdf 
    pdf = norm.pdf(b)   
    if minimization:
        mu = r1.mu * mcdf + r2.mu * cdf - a * pdf 
        var = (r1.mu**2 + r1.var) * mcdf
        var += (r2.mu**2 + r2.var) * cdf
        var -= (r1.mu + r2.mu) * a * pdf
        var -= mu**2 
    else:
        mu = r1.mu * cdf + r2.mu * mcdf + a * pdf      
        var = (r1.mu**2 + r1.var) * cdf
        var += (r2.mu**2 + r2.var) * mcdf
        var += (r1.mu + r2.mu) * a * pdf
        var -= mu**2         
    return RV(mu, var)   

def summary_statistics(data):
    """Compute summary statistics for data."""
    stats = {}
    # Mean.
    stats["MEAN"] = np.mean(data)
    # Variance.
    stats["VAR"] = np.var(data)
    # Max.
    stats["MAX"] = max(data)
    # Min.
    stats["MIN"] = min(data)
    # Median.
    stats["MED"] = np.median(data)
    # Skewness.
    stats["SKEW"] = skew(data)
    # Kurtosis.
    stats["KUR"] = kurtosis(data)   
    return stats
    
def h(mu1, var1, mu2, var2):
    """Helper function for Kamburowski method."""
    alpha = np.sqrt(var1 + var2)
    beta = (mu1 - mu2)/alpha
    cdf_beta = norm.cdf(beta) 
    return mu1*cdf_beta + mu2*(1-cdf_beta) + alpha*norm.pdf(beta)                
                
def funder(X):
    """
    Helper function for Kamburowksi method.
    X is any iterable of RVs, sorted in ascending order of their variance.
    """
    if len(X) == 1:
        return X[0].mu
    elif len(X) == 2:
        return h(X[0].mu, X[0].var, X[1].mu, X[1].var)
    else:
        return h(funder(X[:-1]), 0, X[-1].mu, X[-1].var)

def fover(X):
    """
    Helper function for Kamburowksi method.
    X is any iterable of RVs, sorted in ascending order of their variance.
    """
    if len(X) == 1:
        return X[0].mu
    elif len(X) == 2:
        return h(X[0].mu, X[0].var, X[1].mu, X[1].var)
    else:
        return h(fover(X[:-1]), X[-2].var, X[-1].mu, X[-1].var)  
    
def path_max(P, method="MC", samples=100):
    """
    
    Estimate the maximum of a set of paths.
    
    Parameters
    ----------
    P : TYPE
        DESCRIPTION.
    samples : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    None.
    """
    
    if method == "SCULLI" or method == "S":
        S = P[0].length
        for path in P[1:]:
            S = clark(S, path.length)
        return S
    elif method == "CorLCA" or method == "C":
        dom_path = P[0]
        C = P[0].length
        for path in P[1:]:
            r = path.get_rho(dom_path)
            if path.length.mu > C.mu:
                dom_path = path
            C = clark(C, path.length, rho=r)
        return C
    elif method == "MC" or method == "mc" or method == "M":    
        # Construct vector of means.
        means = [pth.length.mu for pth in P]        
        # Compute covariance matrix.
        cov = []
        for i, pth in enumerate(P):
            row = []
            # Copy already computed covariances.
            row = [cov[j][i] for j in range(i)]
            # Add diagonal - just the variance.
            row.append(pth.length.var)
            # Compute covariance with other paths.
            for pt in P[i + 1:]: 
                rho = pth.get_rho(pt)
                cv = rho * np.sqrt(pth.length.var) * np.sqrt(pt.length.var)
                row.append(cv)
            cov.append(row)        
        # Generate the path length realizations.
        N = np.random.default_rng().multivariate_normal(means, cov, samples)        
        # Compute the maximums.
        dist = np.amax(N, axis=1)        
        return list(dist) # Note this is list rather than RV - make separate function?
    
        
        
            
        
    
    
    
    

    