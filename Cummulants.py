# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:36:54 2024

@author: Fredrik Bergelv
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd 

x = np.linspace(0, 2, 1000)

def f(x):
    """ Function we want to distribute over interval, else 0 """
    
    function = np.pi/2 * np.sin(np.pi*x) # Value of function
    interval = 0, 1                      # Interval for function
    ifnot = 0                            # If we are outside the interval
    result = np.where((x > interval[0]) & (x < interval[1]), 
                      function, ifnot)
    return result

def Accept_Reject(x, n_samples=50):
    """ Accept/Reject method to generate samples distributed according to f(x) """
    val_list = []
    def max_f(R):
        return np.pi / 2                        #<--------------------------
    
    def sample(R):
        result = R                              #<--------------------------
        return result
    
    while len(val_list) < n_samples:
        R1 = rd.random() 
        R2 = sample(rd.random())  
        
        # Accept if R1 < f(R2) / max(f(x)) (i.e., scaled acceptance criterion)
        if R1 < f(R2) / max_f(R2):
            val_list.append(R2)
    
    return np.array(val_list)

def C(X, n_samples=50):
    """ Transformation method, counts values in C_array and finds normalized frequency """
    C_list = []
    
    while len(C_list) < n_samples:
        R = rd.random()  
        C_function = (1/np.pi) * np.arccos(1 - 2*R)          #<----------------
        C_list.append(C_function)
    
    return np.array(C_list)



plt.plot(x,f(x),label='Original function', c='black')

transformation_method = C(x)
plt.scatter(transformation_method, f(transformation_method), label='transformation method', c='b' )

Accept_Reject = Accept_Reject(x)
plt.scatter(Accept_Reject, f(Accept_Reject), label='Accept/Reject method', marker='x', c='r')



plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()