#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This file represents a basic TSP (Travel Salesman Problem)
and uses different optimization algorithms to solve it 
@author: Fernando Lopez Hernandez """

import random
import numpy as np
import exhaustive
import montecarlo
import local

TOO_MUCH = 100.0
weights = [[     0.0, 1.0, 4.0, TOO_MUCH],
           [     1.0, 0.0, 3.0,      2.0],
           [     4.0, 3.0, 0.0,      5.0],
           [TOO_MUCH, 2.0, 5.0,      0.0]]

def permutations(n):
    """ Returns a list with the permutations of [1..n] """
    if (n==1):
        return [[1]]
    base = permutations(n-1)
    extended = []
    for i in range(len(base)):
        for j in range(len(base[i]),-1,-1):
            extended.append(base[i][:j]+[n]+base[i][j:])
    return extended

def tsp_cost(sol):
    """ Return the cost of a solution to the TSP """
    # Sum the weights 
    w = sum([weights[sol[i]-1][sol[i+1]-1] for i in range(len(sol)-1)])
    # If s is not a permutation, add one TOO_MUCH for each repetition
    uniques = np.unique(sol)
    w += TOO_MUCH * (len(sol)-len(uniques))
    return w

def tsp_neighbors(domains, sol):
    """ Return the neighbors of sol
    computes as all the feasible exchanges between two variables """
    neighbors = []
    for i in range(len(sol)):
        for j in range(i+1,len(sol)):
            neighbor = sol.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

space = permutations(4)
domains = [(1,4)] * 4
        
res = exhaustive.fss(space, tsp_cost)
print ("fss cost:", res[0], "sol:", res[1])

res = exhaustive.fds(domains, 1, tsp_cost)
print ("fds cost:", res[0], "sol:", res[1])

res = montecarlo.rs(domains, tsp_cost, 1000, True)
print ("rs cost:", res[0], "sol:", res[1])

start_sol = list(range(1,5))
while (tsp_cost(start_sol) > 2*TOO_MUCH):
    random.shuffle(start_sol)
res = local.hc(start_sol, domains, lambda sol : -tsp_cost(sol),
               tsp_neighbors, max_it = 10000)
print ("hc profit:", res[0], "sol:", res[1])

res = local.sa(start_sol, domains, tsp_cost, tsp_neighbors)
print ("sa cost:", res[0], "sol:", res[1])

res = local.ts(start_sol, domains, lambda sol : -tsp_cost(sol), tsp_neighbors,
               stop_profit=-7, max_it= 10000)
print ("ts profit:", res[0], "sol:", res[1])
