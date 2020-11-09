#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize task timing data.
"""

import dill, pathlib, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
from timeit import default_timer as timer

nbs = [128, 1024]
kernels = ["GEMM", "POTRF", "SYRK", "TRSM"]

with open("summarize_task_timings.txt", "w") as dest:
    print("SUMMARY OF TIMINGS FOR CHOLESKY FACTORIZATION BLAS/LAPACK KERNELS ON CPU AND GPU.", file=dest) 
    for nb in nbs:
        print("\n\n\nTILE SIZE: {}".format(nb), file=dest)
        with open('skylake_V100_samples/no_adt_nb{}.dill'.format(nb), 'rb') as file:
            comp_data, comm_data = dill.load(file)
        for p in ["C", "G"]:
            p_type = "CPU" if p == "C" else "GPU"
            print("\nPROCESSOR TYPE: {}".format(p_type), file=dest)
            for kernel in kernels:
                print("{}: mu = {}, sigma = {}".format(kernel, np.mean(comp_data[p][kernel]), np.std(comp_data[p][kernel])), file=dest)