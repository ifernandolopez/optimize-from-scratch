# -*- coding: utf-8 -*-
""" This file implements exhaustive search approaches 
@author: Fernando Lopez Hernandez """

import numpy as np

def fss(space, cost_fn):
    """ Full space search method 
    space - solutions x variables matrix with all the  solution satisfying the contraints
    cost_fn - the cost function
    Return a tuple (best_cost,best_sol) with the cost of the best (minimum cost) solution in space """
    best_sol = None
    best_cost = np.inf
    for sol in space:
        if (cost_fn(sol) < best_cost):
            best_sol = sol
            best_cost = cost_fn(sol)
    return (best_cost, best_sol)

def fds(domains, step, cost_fn, partial_sol=[]):
    """ Depth-first search method
    Evaluate all the domain assigments 
    domains - The domain of each variable
              Its length determine the number of pending variables to expand 
    step - the granularity of the values of the variables; 1 for discrete variables
    Return a tuple (best_cost,best_sol) for the best variables assigment """
    assert(len(domains) >= 1)
    (best_total_sol, best_cost) = (None, np.inf)
    (first_domain, rest_domains) = (domains[0], domains[1:]) 
    expanded_sols = [] # Descend a level by expanding solutions
    for v in np.arange(first_domain[0], first_domain[1]+step, step):
        expanded_sols.append(partial_sol + [v])
    for sol in expanded_sols: # For each expanded solution
        if (len(rest_domains) == 0): # Compute the leave cost
            (cost, total_sol) = (cost_fn(sol), sol)
        else: # Go through sub branches
            (cost, total_sol) = fds(rest_domains, step, cost_fn, sol)
        if (cost < best_cost):  # Select the best leave
            (best_total_sol, best_cost) = (total_sol, cost)
    return (best_cost, tuple(best_total_sol))
