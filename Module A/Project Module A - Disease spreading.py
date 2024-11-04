"""
Created on Mon Sep 10 2024
@author: Fredrik Bergelv
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

b = np.mean([2.8,3.5])/np.mean([1,7])   # Rate of infection 
m = np.mean([30,75])/100/np.mean([1,7]) # Rate of mortality
g = 0.0#1*2                             # Rate of vaccination 
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


def SIR(t, f): 
    N=300000
    I, S, R = f
    dIdt = b*S*I/N-a*I
    dSdt = -b*S*I/N-g*S
    dRdt = g*S+a*I   
    return dIdt, dSdt, dRdt

def SIR_travel(t, f): 
    I_L, S_L, R_L, I_S, S_S, R_S = f
    
    g_L, g_S = g, g

    dI_L =  b*S_L*I_L/N_L - a*I_L + (w_SL*I_S-w_LS*I_L)
    dS_L = -b*S_L*I_L/N_L - g_L*S_L + (w_SL*S_S-w_LS*S_L)
    dR_L =  g_L*S_L + a*I_L + (w_SL*R_S-w_LS*R_L)
    
    dI_S =  b*S_S*I_S/N_S - a*I_S + (w_LS*I_L-w_SL*I_S)
    dS_S = -b*S_S*I_S/N_S - g_S*S_S + (w_LS*S_L-w_SL*S_S)
    dR_S =  g_S*S_S + a*I_S + (w_LS*R_L-w_SL*R_S)

    return dI_L, dS_L, dR_L, dI_S, dS_S, dR_S


def SIRD_travel(t, f): 
    I_L, S_L, R_L, M_L, I_S, S_S, R_S, M_S = f
     
    if I_L > 100 or I_S > 100:
        g_L = g
        g_S = g
    else: 
        g_L = 0
        g_S = 0

    dI_L =  b*S_L*I_L/N_L - a*I_L - m*I_L + (w_SL*I_S-w_LS*I_L)
    dS_L = -b*S_L*I_L/N_L - g_L*S_L + (w_SL*S_S-w_LS*S_L)
    dR_L =  g_L*S_L + a*I_L + (w_SL*R_S-w_LS*R_L)
    dM_L =  m*I_L
    
    dI_S =  b*S_S*I_S/N_S - a*I_S - m*I_S + (w_LS*I_L-w_SL*I_S)
    dS_S = -b*S_S*I_S/N_S - g_S*S_S + (w_LS*S_L-w_SL*S_S)
    dR_S =  g_S*S_S + a*I_S + (w_LS*R_L-w_SL*R_S)
    dM_S =  m*I_S
    
    return dI_L, dS_L, dR_L, dM_L, dI_S, dS_S, dR_S, dM_S

#%% SIR
InitialCond1 = I0_L, S0_L, R0_L, I0_S, S0_S, R0_S

sol1 = solve_ivp(SIR_travel, [t0, t1], InitialCond1, 
                method='RK45', t_eval=np.linspace(t0, t1, h))

for i in range(0,6):
    sol1.y[i]=sol1.y[i]/1000
I_L, S_L, R_L = sol1.y[0], sol1.y[1], sol1.y[2]
I_S, S_S, R_S = sol1.y[3], sol1.y[4], sol1.y[5]
t = sol1.t

plt.figure(figsize=(15*0.6, 6*0.6))

plt.subplot(1, 2, 1)
plt.plot(t, I_L, label='Infected')
plt.plot(t, S_L, label='Susceptible')
plt.plot(t, R_L, label='Recovered')
plt.xlabel('Time [days]')
plt.ylabel('Population [thousands]')
plt.title(f'Disease spread in Lübeck with $\gamma={int(g*100)}\%$')
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


def SIR_info(sol1):
    for i in range(0,5):
        sol1.y[i]=sol1.y[i]/1000
        I_L, S_L, R_L = sol1.y[0], sol1.y[1], sol1.y[2]
        I_S, S_S, R_S = sol1.y[3], sol1.y[4], sol1.y[5]
   
    for i in range (0,h):
        if I_L[i]==np.max(I_L):
            I_L_max = int(np.round(100*I_L[i]/(I_L[i]+S_L[i]+R_L[i])))
        if I_S[i]==np.max(I_S):
            I_S_max = int(np.round(100*np.max(I_S)/(I_S[i]+S_S[i]+R_S[i])))
        
        Imune = int(100*R_S[-1]/(S_S[-1]+R_S[-1]))

    return print(f'SIR: The simulation used g = {int(g*100)}%. The a maximal fraction of infected in Lübeck was {I_L_max}% and maximal fraction of infected in Scania was {I_S_max}%. After the plauge {Imune}% of people are imune.')

SIR_info(sol1)

sol_h = solve_ivp(SIR_travel, [t0, t1], InitialCond1, 
                method='RK45', t_eval=np.linspace(t0, t1, h))
sol_2h = solve_ivp(SIR_travel, [t0, t1], InitialCond1, 
                method='RK45', t_eval=np.linspace(t0, t1, h*2))

Titles = ('Infected in Lübeck', 'Susceptible in Lübeck', 
          'Recovered in Lübeck','Infected in Scania', 
          'Susceptible in Scania','Recovered in Scania')

m_err = 4
plt.figure(figsize=(7*0.8/0.5*0.6, 5*0.8/0.5*0.6))
for i in range(0,6):
    Err = (sol_h.y[i]-np.interp(sol_h.t,sol_2h.t,sol_2h.y[i]))/(2**m_err-1)
    plt.plot(sol_h.t, Err, label=Titles[i])
plt.xlabel('Time [days]')
plt.ylabel('Absolute error')
plt.title('Absolute error in disease spread')
plt.legend()
plt.grid(True)
plt.legend(loc='lower right')

plt.tight_layout()

#%% SIRD
InitialCond2 = I0_L, S0_L, R0_L, M0_L, I0_S, S0_S, R0_S, M0_S

sol2 = solve_ivp(SIRD_travel, [t0, t1], InitialCond2, 
                method='RK45', t_eval=np.linspace(t0, t1, h))

for i in range(0,8):
    sol2.y[i]=sol2.y[i]/1000
I_L, S_L, R_L, M_L = sol2.y[0], sol2.y[1], sol2.y[2], sol2.y[3]
I_S, S_S, R_S, M_S = sol2.y[4], sol2.y[5], sol2.y[6], sol2.y[7]
t = sol2.t

plt.figure(figsize=(15*0.6, 6*0.6))

plt.subplot(1, 2, 1)
plt.plot(t, I_L, label='Infected')
plt.plot(t, S_L, label='Susceptible')
plt.plot(t, R_L, label='Recovered')
plt.plot(t, M_L, label=f'Deaths = {round(100*np.max(M_L*1e3)/N_L)}%')
plt.xlabel('Time [days]')
plt.ylabel('Population [thousands]')
plt.title(f'Disease spread in Lübeck with $\gamma={int(g*100)}\%$')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(t, I_S, label='Infected')
plt.plot(t, S_S, label='Susceptible')
plt.plot(t, R_S, label='Recovered')
plt.plot(t, M_S, label=f'Deaths = {round(100*np.max(M_S*1e3)/N_S)}%')
plt.xlabel('Time [days]')
plt.ylabel('Population [thousands]')
plt.title(f'Disease spread in Scania with $\gamma={int(g*100)}\%$')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()


def SIRD_info(sol2):
    for i in range(0,8):
        sol2.y[i]=sol2.y[i]/1000
    I_L, S_L, R_L, M_L = sol2.y[0], sol2.y[1], sol2.y[2], sol2.y[3]
    I_S, S_S, R_S, M_S = sol2.y[4], sol2.y[5], sol2.y[6], sol2.y[7]
    
    for i in range (0,h):
        if I_L[i]==np.max(I_L):
            I_L_max = int(np.round(100*I_L[i]/(I_L[i]+S_L[i]+R_L[i])))
        if I_S[i]==np.max(I_S):
            I_S_max = int(np.round(100*np.max(I_S)/(I_S[i]+S_S[i]+R_S[i])))
        
        M_S_end = int(np.round(100*1000*(+M_S[-1])/(N_S)))
        M_L_end = int(np.round(100*1000*(+M_L[-1])/(N_L)))
        M_end = int(np.round(100*1000*(M_L[-1]+M_S[-1])/(N_S+N_L)))
        Imune = int(100*R_S[-1]/(S_S[-1]+R_S[-1]))

    return print(f'SIRD: The simulation used g = {int(g*100)}%. This led to a total of {M_L_end}% deaths in Lübeck and {M_S_end}% in Scania with a total death toll of {M_end}%. The a maximal fraction of infected in Lübeck was {I_L_max}% and maximal fraction of infected in Scania was {I_S_max}%. After the plauge {Imune}% of people are imune.')

SIRD_info(sol2)

#%%