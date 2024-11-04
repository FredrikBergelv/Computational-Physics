"""
Created on Mon Sep 10 2024
@author: Fredrik Bergelv
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

b = np.mean([2.8,3.5])/np.mean([1,7])   # Rate of infection 
m = np.mean([30,75])/100/np.mean([1,7]) # Rate of mortality
g = 0.01*2                              # Rate of vaccination 
a = 1/np.mean([1,7])-m                  # Rate of recovery

N_L = 38000  
N_S = 150000  
N_ship = 13
Number_ships_per_day = 2
w_SL = Number_ships_per_day*N_ship/N_S  
w_LS = Number_ships_per_day*N_ship/N_L 

I0_L, S0_L, R0_L, M0_L = N_ship, N_L-N_ship, 0, 0 
I0_S, S0_S, R0_S, M0_S = 0, N_S, 0, 0  
timescale = t0, t1 = 0, 31*4
h = 10000

InitialCond = I0_L, S0_L, R0_L, I0_S, S0_S, R0_S


def SIR_travel(t, f): 
    I_L, S_L, R_L, I_S, S_S, R_S = f
    
    g_L = g
    g_S = g
    
    dI_L =  b*S_L*I_L/N_L - a*I_L + (w_SL*I_S-w_LS*I_L)
    dS_L = -b*S_L*I_L/N_L - g_L*S_L + (w_SL*S_S-w_LS*S_L)
    dR_L =  g_L*S_L + a*I_L + (w_SL*R_S-w_LS*R_L)
    
    dI_S =  b*S_S*I_S/N_S - a*I_S + (w_LS*I_L-w_SL*I_S)
    dS_S = -b*S_S*I_S/N_S - g_S*S_S + (w_LS*S_L-w_SL*S_S)
    dR_S =  g_S*S_S + a*I_S + (w_LS*R_L-w_SL*R_S)

    
    return dI_L, dS_L, dR_L, dI_S, dS_S, dR_S


sol = solve_ivp(SIR_travel, [t0, t1], InitialCond, 
                method='RK45', t_eval=np.linspace(t0, t1, h))

for i in range(0,6):
    sol.y[i]=sol.y[i]/1000
I_L, S_L, R_L = sol.y[0], sol.y[1], sol.y[2]
I_S, S_S, R_S = sol.y[3], sol.y[4], sol.y[5]
t = sol.t

plt.figure(figsize=(15*0.6, 6*0.6))

plt.subplot(1, 2, 1)
plt.plot(t, I_L, label='Infected')
plt.plot(t, S_L, label='Susceptible')
plt.plot(t, R_L, label='Recovered')
plt.xlabel('Time [days]')
plt.ylabel('Population [thousands]')
plt.title(f'Disease spread in Lübeck with $\gamma={int(g*100)}\%$')
plt.legend()
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(t, I_S, label='Infected')
plt.plot(t, S_S, label='Susceptible')
plt.plot(t, R_S, label='Recovered')
plt.xlabel('Time [days]')
plt.ylabel('Population [thousands]')
plt.title(f'Disease spread in Scania with $\gamma={int(g*100)}\%$')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()


for i in range (0,h):
    if I_L[i]==np.max(I_L):
        I_L_max = int(np.round(100*I_L[i]/(I_L[i]+S_L[i]+R_L[i])))
    if I_S[i]==np.max(I_S):
        I_S_max = int(np.round(100*np.max(I_S)/(I_S[i]+S_S[i]+R_S[i])))
        

Imune = int(100*R_S[-1]/(S_S[-1]+R_S[-1]))

print(f'Imune are {Imune}% and {I_L_max}% are infected at once in Lubeck and {I_S_max}% in Scania')

