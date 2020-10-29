#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summaries and plots for the empirical distributions.
"""

import dill, pathlib, sys
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import shapiro, skew, kurtosis

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

####################################################################################################

# Destinations to save summaries and generated plots.
summary_path = "summaries/empirical_dists"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/empirical_dists"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

nb = 128
n_tasks = [35, 220, 680, 1540, 2925, 4960, 7770, 11480]
# Load info.
with open('empirical_dists.dill', 'rb') as file:
    empirical = dill.load(file)     

# =============================================================================
# Readable summaries of tests of distribution.
# =============================================================================

with open("{}/summary_statistics.txt".format(summary_path), "w") as dest:
    print("TESTING THE NORMALITY ASSUMPTION OF THE EMPIRICAL DISTRIBUTION FOR THE CHOLESKY DAG SET.", file=dest) 
    print("EMPIRICAL DISTRIBUTIONS GENERATED THROUGH 100,000 REALIZATIONS OF ENTIRE GRAPH.", file=dest)
    
    for nt in n_tasks:
        print("\n\n\n---------------------------------", file=dest)
        print("NUMBER OF TASKS: {}".format(nt), file=dest)
        print("---------------------------------", file=dest)
        for dist in ["NORMAL", "GAMMA"]: 
            print("{} WEIGHTS".format(dist), file=dest)
            mu = np.mean(empirical[nt][dist])
            print("\nMEAN: {}".format(mu), file=dest)
            var = np.var(empirical[nt][dist])
            print("VARIANCE/STD: {} / {}".format(var, np.sqrt(var)), file=dest)
            skw = skew(empirical[nt][dist])
            print("SKEWNESS: {}".format(skw), file=dest)
            kur = kurtosis(empirical[nt][dist])
            print("EXCESS KURTOSIS: {}".format(kur), file=dest)
            # stat, p = shapiro(empirical[nt][dist])
            # verdict = "NORMAL" if p > 0.05 else "NOT NORMAL"
            # print("p = {} - {}".format(p, verdict), file=dest)
            med = np.median(empirical[nt][dist])
            print("MEDIAN: {}".format(med), file=dest)
            mx = max(empirical[nt][dist])
            print("MAXIMUM: {}".format(mx), file=dest)
            mn = min(empirical[nt][dist])
            print("MINIMUM: {}".format(mn), file=dest)
            
        

            