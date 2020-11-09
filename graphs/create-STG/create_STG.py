#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create and save STG DAGs.
"""

import dill, pathlib, os, re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
from timeit import default_timer as timer
from scipy.stats import norm
import sys
sys.path.append('../../')   
from src import SDAG

src = 'original'
dest ='../STG'

# =============================================================================
# RV class.
# =============================================================================

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
    
# =============================================================================
# Create and save the DAGs.  
# =============================================================================

start = timer()
# Read stg files.
s = 0
for orig in os.listdir(src):    
    if orig.endswith('.stg'):  
        print("\n{}".format(orig))
        s += 1
        G = nx.DiGraph()       
        with open("{}/{}".format(src, orig)) as f:
            next(f) # Skip first line.            
            for row in f:
                if row[0] == "#":                   
                    break
                # Remove all whitespace - there is probably a nicer way to do this...
                info = " ".join(re.split("\s+", row, flags=re.UNICODE)).strip().split() 
                # Create task.   
                nd = RV()
                nd.ID = int(info[0])
                if info[2] == '0':
                    G.add_node(nd)
                    continue
                # Add connections to predecessors.
                predecessors = list(n for n in G if str(n.ID) in info[3:])
                for p in predecessors:
                    G.add_edge(p, nd)                  
        
        # Convert G to an SDAG object. 
        S = SDAG(G) 
        
        for c in [0.1, 0.3, 0.5]:
            for d in [0.1, 0.3, 0.5]:
                # Set weights.
                S.set_weights(c, d)                           
                # Save DAG.
                with open('{}/T{}_C{}_D{}.dill'.format(dest, s, c, d), 'wb') as handle:
                    dill.dump(S, handle)    
        
elapsed = timer() - start     
print("Time taken: {} seconds".format(elapsed))  
