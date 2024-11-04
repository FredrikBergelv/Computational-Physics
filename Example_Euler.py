"""
Created on Thu Feb 22 15:43:32 2024
@author: Fredrik Bergelv
"""
import numpy as np
import matplotlib.pyplot as plt
from  uncertainties import ufloat 
#import random as rd 
#import scipi as sci 
#import pandas as ps


def p(w):
    py=np.exp(-w)
    return py

Yt = np.linspace(0,4,100)
Pt = p(Yt)

lenght = 300

Pe = np.zeros((2, lenght-1))

def euler(h):
    for n in range(0,lenght-1):
        h = h
        y = n*h
        if y>4:
            break
        Pe[0,n]=y
        if n == 0:
            Pe[1,n] = 1
        else : 
            Pe[1,n]= Pe[1,n-1]*(1-h)
    P = Pe[1]
    Y = Pe[0]
    return plt.scatter(Y,P, label=f'h=1/{(i*2)}')

for i in range(1,5):
    h=1/(i*2)
    if i==3:
        0
    else: 
        euler(h)

plt.plot(Yt,Pt, 'black', label='theory')
plt.xlabel(r'$y/l_0$')
plt.ylabel(r'$\rho/\rho_0$')
plt.xlim(0,4)  
plt.legend()
plt.show()
