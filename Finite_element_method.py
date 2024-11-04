# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:44:59 2024

@author: Fredrik Bergelv
"""

import numpy as np
import matplotlib.pyplot as plt

# Definition of the local basis functions
def local_basis_1(xi):
    return 0.5*(1-xi)

def local_basis_2(xi):
    return 0.5*(1+xi)

# Define a mapping from global to local parametization
def global_to_local(x, xminus, xplus):
    return (2*x-xminus-xplus) / (xplus-xminus)

# Define the global basis function for node i
def global_basis(x, nodes, i):
    
    if i == 0:
        if x >= nodes[0] and x <= nodes[1]:
            xi = global_to_local(x, nodes[0], nodes[1])
            return local_basis_1(xi)
        else:
            return 0
        
    elif i == len(nodes) - 1:
        if x >= nodes[-2] and x <= nodes[-1]:
            xi = global_to_local(x, nodes[-2], nodes[-1])
            return local_basis_2(xi)
        else:
            return 0
    
    else:
        if x >= nodes[i-1] and x <= nodes[i]:
            xi = global_to_local(x, nodes[i-1], nodes[i])
            return local_basis_2(xi)
        elif x >= nodes[i] and x <= nodes[i+1]:
            xi = global_to_local(x, nodes[i], nodes[i+1])
            return local_basis_1(xi)
        else:
            return 0

# Define the domain and nodes
def plot_basis_functions(N):
    nodes = np.linspace(0, N, N+1)
    x_plot = np.linspace(nodes[0], nodes[-1], 1000)
    
    for i in range(len(nodes)):
        y_plot = [global_basis(x, nodes, i) for x in x_plot]
        plt.plot(x_plot, y_plot, label='Global basis function')
    
    # Plot the nodes
    for node in nodes:
        plt.axvline(x=node, color='gray', linestyle='--')
    
    plt.title('Global Basis Functions')
    plt.xlabel('$x$')
    plt.ylabel('Basis function value')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

#%%

def sin(x):
    return np.sin(x)

# Finite Element Method implementation
def finite_element_method(func, N, a, b, phi=global_basis):
    nodes = np.linspace(a, b, N)  # Nodes
    x_plot = np.linspace(nodes[0], nodes[-1], 1000)  # Points for evaluation

    # Approximate the function f(x) using the basis functions
    func_approx = np.zeros_like(x_plot)  # To store the approximated function

    for i in range(len(nodes)):
        phi_vals = np.array([global_basis(x, nodes, i) for x in x_plot])
        func_val_at_node = func(nodes[i])  # Function value at node i
        func_approx += func_val_at_node * phi_vals  # Linear combination

    return x_plot, func_approx

N = 15
a,b = 0,10  

x_vals, approx_vals = finite_element_method(sin, N, a, b)

plt.plot(x_vals, np.sin(x_vals), label='$\sin(x)$', color='blue')
plt.plot(x_vals, approx_vals, label='FEM approximation', color='red', linestyle='--')
plt.title('FEM Approximation of $\sin(x)$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


#%%

plot_basis_functions(N)

