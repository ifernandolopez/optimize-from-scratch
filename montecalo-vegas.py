#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Comparison between Monte Carlo and Las Vegas random search algorithm
@author: Fernando Lopez Hernandez """

import numpy as np

def montecarlo_search(value, A, n):
    """ Return the index in A of the most approximate value
    sampling n times """
    best_k = -1
    best_dist = np.inf
    for i in range(n):
        k = np.random.randint(0,len(A)-1)
        if (abs(A[k]-value) < best_dist):
            best_k = k
            best_dist = abs(A[k]-value)
    return best_k

def las_vegas(value, A):
    while True:
        k = np.random.randint(0,len(A)-1)
        if (A[k] == value):
            return k

N = 10000000
n = 100000
A = np.random.randint(0,N,n)
value = A[n//2]

k = montecarlo_search(value,A, n)
print("Monte Carlo k:", k, "A[k]:", A[k])
k = las_vegas(value, A)
print("Las Vegas k:", k, "A[k]:", A[k])