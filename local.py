# -*- coding: utf-8 -*-
""" This file implements local search algorithms
@author: Fernando Lopez Hernandez """

import math
import random
import numpy as np
import collections

def accuracy_ratio(best_profit, achieved_profit):
    """ Quantifies the accuracy of the approximate solution """
    return achieved_profit/best_profit

def profitable_neighbor(sol, domains, neighbors_fn, profit_fn):
    """ Helper function that returns the best profit and neighbor
    This function is used by several local search algorithms
    neighbors_fn - function that determine the neighbor solutions
    profit_fn - therefore cost should be negated """
    best_n, best_profit = None, -np.inf
    neighbors = neighbors_fn(domains, sol)
    for n in neighbors:
        n_profit = profit_fn(n)
        if (n_profit > best_profit):
            best_profit, best_n = n_profit, n
    return (best_profit, best_n)


def hc(start_sol, domains, profit_fn, neighbors_fn, max_it):
    """ Implement hill climbing
    start_sol - initial solution
    profit_fn - the profit funcion
    beighbors_fn - the neighbors function
    max_it - maximum number of iterations """
    sol = start_sol
    current_profit = profit_fn(sol)
    for i in range(max_it):
        # Search for the more profitable neighbor
        best_n, best_profit = None, -np.inf
        neighbors = neighbors_fn(domains, sol)
        for n in neighbors:
            n_profit = profit_fn(n)
            if (n_profit > best_profit):
                best_profit, best_n = n_profit, n
        # If there is no improvement, then we have reach the top
        if (best_profit <= current_profit):
            break
        else:
            sol, current_profit = best_n, best_profit
    return (current_profit, sol)

def estimated_gradient(x, cost_fn, h):
    """ Return the estimated gradient when cost_fn is nod differentiable
    x - current sol vector
    h - advance """
    g = []
    for i in range(len(x)):
        # Calculate x+h for the i-th variable
        x_h = x.copy()
        x_h[i] = x[i] + h
        # Calculate the i-th partial derivate
        diff = (cost_fn(x_h) - cost_fn(x)) / h
        g.append(diff)
    return g

def dgd(start_sol, cost_fn, gradient_fn, tolerance = 0.00001):
    """ Implements depth gradient descent
    start_sol - initial sol
    tolerance - minimum step to keep searching """
    sol = np.array(start_sol)
    cost = cost_fn(sol)
    while True:
        direction = -1 * np.array(gradient_fn(sol))
        next_sol = sol + 0.01*direction
        next_cost = cost_fn(next_sol)
        # Convergence condition
        if (abs(cost-next_cost) <= tolerance):
            break
        sol = next_sol
        cost = next_cost
    return (cost, sol)

def bgd(start_sol, cost_fn, gradient_fn, advances, tolerance = 0.00001):
    """ Implements breadth gradient descent
    start_sol - initial sol
    advances - distances to evaluate in each step 
    tolerance - minimum step to keep searching """
    sol, cost = np.array(start_sol), cost_fn(start_sol)
    while True:
        direction = -1 * np.array(gradient_fn(sol))
        # Breadth search the best next sol
        (best_next_sol, best_next_cost) = sol, np.inf
        for a in advances:
            next_sol = sol + a*direction
            next_cost = cost_fn(next_sol)
            if (best_next_cost > next_cost):
                (best_next_sol, best_next_cost) = next_sol, next_cost
        # Convergence condition
        if (cost-best_next_cost <= tolerance):
            break
        # Make an in depth step
        (sol, cost) = (best_next_sol, best_next_cost)
    return (best_next_cost, sol)

def sa(start_sol, domains, cost_fn, neighbors_fn, T=100000.0, cool_factor = 0.999):
    """ Implements simulated annealing
    neightbors_fn - function that determines the neighbors
    T - initial temperature
    cool_factor - according to the formula t = cool_factor*t """
    best_sol = sol = np.array(start_sol)
    best_E = Ea = cost_fn(start_sol)
    while (T>0.01):
        # Choice a random neighbor sol
        neighbors = neighbors_fn(domains, sol)
        next_sol = random.choice(neighbors)
        # Calculate next energy
        Eb = cost_fn(next_sol)
        # Update sol if next_sol has lower cost (p>1)
        # or we pass the probability cutoff
        p =pow(math.e, (Ea-Eb)/T)
        if (p > 1.0 or p > np.random.uniform()):
            sol = next_sol
            Ea = Eb
            # Save the best ever found
            if (Eb < best_E):
                best_sol = next_sol
                best_E = Eb
        # Decrease temperature
        T = cool_factor * T
    return (best_E, list(best_sol))

def ts(start_sol, domains, cost_fn, neighbors_fn, stop_cost, max_it = 10000, max_tl_len = 100):
    tl = [start_sol]
    best_sol = current_candidate = start_sol
    best_cost = cost_fn(start_sol)
    for i in range(max_it):
        # Each iteration choses a neighbor of current_candidate
        neighbors = neighbors_fn(domains, current_candidate)
        # First tries to choose randomly an un-vetoed candidate: not in the tl
        unvetoed_neighbors = [candidate for candidate in neighbors if candidate not in tl]
        if len(unvetoed_neighbors) > 0:
            next_candidate = random.choice(unvetoed_neighbors)
            next_candidate_cost = cost_fn(next_candidate)
        # Otherwise uses the aspiration criteria and chooses the best vetoed neighbor
        else:
            # next_candidate = max(neighbors, key = cost_fn)
            # next_candidate_cost = cost_fn(next_candidate)
            next_candidate, next_candidate_cost = None, np.inf
            for candidate in neighbors:
                candidate_cost = cost_fn(candidate)
                if candidate_cost < next_candidate_cost:
                    next_candidate, next_candidate_cost = candidate, candidate_cost
        # Update the best_sol, if a better candidate is found
        if next_candidate_cost < best_cost:
            best_sol, best_cost = next_candidate, next_candidate_cost
        # Anyway, update the current_candidate
        current_candidate = next_candidate
        # If we have reach the stop_cost
        if best_cost <= stop_cost:
            break
        # Veto the candidate
        tl.append(next_candidate)
        # Limit the size of the tl
        if len(tl) > max_tl_len:
            tl = tl[len(tl)//2:]
    return (best_cost, best_sol)

    
