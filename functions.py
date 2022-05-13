#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This file represents a few well-known problems: n-dimensions and continious variable 
to be used in the optimization algorithms
The file also cointains general purpose functions
@author: Fernando López Hernández """

import math
import numpy as np
import itertools
import exhaustive
import montecarlo
import local

def sphere_cost(sol):
    """ Return the cost of the shpere problem """
    return sum([v**2.0 for v in sol])

def sphere_gradient(sol):
    """ Return the gradient of the shere problem """
    return [2.0*v for v in sol]

def grid_neighbors(domains, sol, step = 0.1):
    " Return the neighbors solutions in the grid"
    dimensions = len(sol)
    neighbors = []
    for d in range(dimensions):
        if (sol[d]+step <= domains[d][1]):
            upper_sol = sol.copy()
            upper_sol[d] = sol[d]+step 
            neighbors.append(upper_sol)
        if (sol[d]-step >= domains[d][0]):
            lower_sol = sol.copy()
            lower_sol[d] = sol[d]-step
            neighbors.append(lower_sol)
    return neighbors

def rastigin_cost(sol):
    """ Return the cost of the rantrigin problem """
    sol = np.array(sol)
    return 10.0*len(sol) + sum(sol**2 - 10*np.cos(2*math.pi*sol))

def rastigin_gradient(sol):
    """ Return the gradient of the rastrigin problem """
    return [2.0*v + (2*math.pi) * math.sin(2*math.pi*v) for v in sol]

def cartesian(start, stop, step, d):
    """ Return the cartesian product of d sampled variables """
    var_samples = np.arange(start, stop+step, step)
    vars_samples = [var_samples.tolist() for _ in range(d)]
    return [sol for sol in itertools.product(*vars_samples)]

def rnd(x):
    if (isinstance(x, (tuple,list))):
        return tuple([round(i,2) for i in x])
    if (isinstance(x, np.ndarray)):
        return tuple(x.round(2))
    return round(x,6)

def print_solution(name, res):
    worst_cost = sphere_cost([5.0]*dimensions)
    print (name, 
           "ratio:", rnd(local.accuracy_ratio(worst_cost, worst_cost-res[0])), 
           "cost:", rnd(res[0]),
           "sol:", rnd(res[1]))

dimensions = 3
subdivision = 2
step = 1.0/subdivision
space = cartesian(-5.0,5.0,step,dimensions)
domains = [(-5.0,5.0)] * dimensions
start_sol = [np.random.uniform(domain[0],domain[1])for domain in domains]
gd_advances = [10.0, 5.0, 1.0, 0.5, 0.1, 0.01, 0.001]


if (__name__ == "__main__"):
    res = exhaustive.fss(space, sphere_cost)
    print_solution("Sphere fss", res)

    res = exhaustive.fds(domains, step, sphere_cost)
    print_solution("Sphere fds", res)

    res = exhaustive.fss(space, rastigin_cost)
    print_solution("Rastigin fss", res)

    res = exhaustive.fds(domains, step, rastigin_cost)
    print_solution("Rastigin fds", res)

    res = montecarlo.rs(domains, sphere_cost, 100000, False)
    print_solution("Sphere rs", res)

    res = montecarlo.rs(domains, rastigin_cost, 100000, False)
    print_solution("Rastigin rs", res)

    res = local.dgd(start_sol, sphere_cost, sphere_gradient)
    print_solution("Sphere dgd", res)

    res = local.bgd(start_sol, sphere_cost, sphere_gradient, gd_advances)
    print_solution("Sphere bgd", res)

    res = local.dgd(start_sol, rastigin_cost, rastigin_gradient)
    print_solution("Rantingin dgd", res)

    res = local.bgd(start_sol, rastigin_cost, rastigin_gradient, gd_advances)
    print_solution("Rantingin bgd", res)

    res = local.sa(start_sol, domains, sphere_cost, grid_neighbors)
    print_solution("Sphere sa", res)

    res = local.sa(start_sol, domains, rastigin_cost, grid_neighbors)
    print_solution("Rastingin sa", res)

    res = local.ts(start_sol, domains, sphere_cost, grid_neighbors, stop_cost = 0, max_it=100000)
    print_solution("Sphere ts", res)

    res = local.ts(start_sol, domains, rastigin_cost, grid_neighbors, stop_cost = 0, max_it=100000)
    print_solution("Rastingin ts", res)
