# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:22:26 2024

@author: Fredrik Bergelv
"""

import numpy as np
import matplotlib.pyplot as plt
from  uncertainties import ufloat 
#import random as rd 
#import scipi as sci 
#import pandas as ps
#plt.savefig(r'C:\Users\fredr\OneDrive\Dokument\Python\plot.pdf')


def f(x):
    return 2/(1+x**2)

I1 = 0.5*(f(0)/2+f(0.5)+f(1)/2)
I2 = 0.25*(f(0)/2+f(0.25)+f(0.5)+f(0.75)+f(1)/2)

print(1/3*(4*I2-I1))