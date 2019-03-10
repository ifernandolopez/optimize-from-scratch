#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This file contains root-finding algorithms
@author: Fernando Lopez Hernandez """

from math import sqrt, exp, tan, pi
from numpy import sign, arange
from numpy.random import ranf

def are_close(x1, x2, allowed_error = 0.01):
    return abs(x1 - x2) <= allowed_error

def parabola_example(x):
    """ The solution to this polinomial is x=1 """
    return x**2-2*x+1

def parabola_prime_example(x):
    return 2*x-2

def degree_3_example(x):
    """ The solution of this polinomial x*(x**2+1) are x=[-1, 0, 1] """
    return (x**3+x)

def degree_3_prime_example(x):
    """ The derivative of x**3+x """
    return 3*x**2+1

def degree_4_example(x):
    """ The solutions of this polinomial are -2, -sqrt(2) sqrt(2), 2 """
    return (x+2)*(x+sqrt(2))*(x-sqrt(2))*(x-2)

def exp_example(x):
    return exp(2*x)-x-6

def exp_prime_example(x):
    return 2*exp(2*x)-1

def tan_example(x):
    return tan(x)

def exhaustive(a, b, fn, step = 0.001):
    sols = []
    prev_sign = sign(fn(a))
    for x in arange(a, b, step):
        current_sign = sign(fn(x))
        if (prev_sign!=current_sign):
            sols.append(x)
            prev_sign = current_sign
    return sols

def bisection(a, b, fn, tolerance = 0.001):
    x_mid = (b+a)/2
    if abs(fn(x_mid)) < tolerance:
        return x_mid
    if sign(fn(a)) != sign(fn(x_mid)):
        return bisection(a, x_mid, fn, tolerance)
    else:
        return bisection(x_mid, b, fn, tolerance)

def bisection_iterative(a, b, fn, tolerance = 0.001):
    while True:
        x_mid = (b+a)/2
        if abs(fn(x_mid)) < tolerance:
            break
        if sign(fn(a)) != sign(fn(x_mid)):
            b = x_mid
        else:
            a = x_mid
    return x_mid

def newton_raphson(start_x, fn, fp, tolerance = 0.00001, max_iter = 1000):
    x = start_x
    while (max_iter>0):
        max_iter -= 1
        f_x = fn(x)
        if abs(f_x)<=tolerance: # Solution found
            break
        f_prime = fp(x)
        if abs(f_prime)<tolerance: # Small derivative
            x = x + ranf()
            continue
        x = x - (f_x/f_prime) # Apply the iterative rule
    return x

def secant(a, b, fn, tolerance = 0.0001, max_iter = 1000):
    f_a, f_b = fn(a), fn(b)
    while (max_iter>0):
        max_iter -= 1
        if abs(f_a) < tolerance:
            break
        if abs(f_b-f_a) < tolerance: # Small derivative
            b = b + ranf()
            f_b = fn(b)
            continue
        # Update a, b interval
        prev_a = a
        a = a - (a-b)/(f_a-f_b)*f_a
        b = prev_a
        f_b = f_a
        f_a = fn(a)
    return a

all_sols = exhaustive(-3, 3, degree_4_example)
assert(len(all_sols)==4)
sol = bisection(-3, 3, degree_4_example)
assert(are_close(sol, sqrt(2)))
sol = bisection(1, 5, lambda x: x**2 - 5)
assert(are_close(sol, sqrt(5)))
sol = bisection_iterative(1, 5, lambda x: x**2 - 5)
assert(are_close(sol, sqrt(5)))
sol = newton_raphson(2,parabola_example, parabola_prime_example)
assert(are_close(sol,1.0))
sol = newton_raphson(3, degree_3_example, degree_3_prime_example)
assert(are_close(sol, 0.0))
sol = bisection(0,2,exp_example)
assert(are_close(sol, 0.9708826555544137))
sol = newton_raphson(2,exp_example,exp_prime_example)
assert(are_close(sol, 0.9708826555544137))
sol = secant(2, 4, tan_example)
assert(are_close(sol, pi))
