# -*- coding: utf-8 -*-
""" This file implements Monte Carlo random search 
@author: Fernando Lopez Hernandez """

import random
import numpy as np

def rs(domains, cost_fn, n, discrete):
    """ Returns the best cost and solutionin a random search
    domains - lower and upper value of each variable 
              the length of domains determine the number of variables n 
    cost_fn - The cost function
    n - number of samples to evaluate """
    (best_sol, best_cost, d) = (None, np.inf, len(domains))
    for _ in range(n):
        # Create a discrete or continious random solution with d values
        if (discrete):
            sol = [random.randint(domains[j][0],domains[j][1]) for j in range(d)]
        else:
            sol = [random.uniform(domains[j][0],domains[j][1]) for j in range(d)]
        # And evaluate it
        cost = cost_fn(sol)
        if (cost < best_cost):
            best_sol = sol
            best_cost = cost
    return (best_cost, best_sol)
