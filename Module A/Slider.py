"""
Created on Mon Sep 10 2024
@author: Fredrik Bergelv
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

b = 0.6  # Rate of infection 
a = 0.04 # Rate of recovery
g = 0.01 # Rate of vaccination 
m = 0.06 # Rate of mortality


N_L = 38000  
N_S = 300000  
N_ship = 13
Number_ships_per_day = 2
w_SL = Number_ships_per_day*N_ship/N_S  
w_LS = Number_ships_per_day*N_ship/N_L 
I0_L, S0_L, R0_L, M0_L = N_ship, N_L-N_ship, 0, 0 
I0_S, S0_S, R0_S, M0_S = 0, N_S, 0, 0  
timescale = t0, t1 = 0, 200

InitialCond = I0_L, S0_L, R0_L, M0_L, I0_S, S0_S, R0_S, M0_S

def SIRD_travel(t, f, b, a, g): 
    I_L, S_L, R_L, M_L, I_S, S_S, R_S, M_S = f
    dI_L =  b*S_L*I_L/N_L - a*I_L - m*I_L + (w_SL*I_S-w_LS*I_L)
    dS_L = -b*S_L*I_L/N_L - g*S_L + (w_SL*S_S-w_LS*S_L)
    dR_L =  g*S_L + a*I_L + (w_SL*R_S-w_LS*R_L)
    dM_L =  m*I_L
    
    dI_S =  b*S_S*I_S/N_S - a*I_S - m*I_S + (w_LS*I_L-w_SL*I_S)
    dS_S = -b*S_S*I_S/N_S - g*S_S + (w_LS*S_L-w_SL*S_S)
    dR_S =  g*S_S + a*I_S + (w_LS*R_L-w_SL*R_S)
    dM_S =  m*I_S
    
    return dI_L, dS_L, dR_L, dM_L, dI_S, dS_S, dR_S, dM_S

def plot_SIRD(g):
    sol = solve_ivp(SIRD_travel, [t0, t1], InitialCond, args=(b, a, g), method='RK45', 
                     t_eval=np.linspace(t0, t1, t1))

    I_L, S_L, R_L, M_L = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    I_S, S_S, R_S, M_S = sol.y[4], sol.y[5], sol.y[6], sol.y[7]
    t = sol.t

    ax[0].clear()  
    ax[1].clear()  

    ax[0].plot(t, I_L, label='Infected')
    ax[0].plot(t, S_L, label='Susceptible')
    ax[0].plot(t, R_L, label='Recovered')
    ax[0].plot(t, M_L, label=f'Deaths = {round(100*np.max(M_L)/N_L)}%')
    ax[0].set_xlabel('Time [days]')
    ax[0].set_ylabel('Population')
    ax[0].set_ylim(0, N_L)
    ax[0].set_title(f'Disease spread in Lübeck')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)

    ax[1].plot(t, I_S, label='Infected')
    ax[1].plot(t, S_S, label='Susceptible')
    ax[1].plot(t, R_S, label='Recovered')
    ax[1].plot(t, M_S, label=f'Deaths = {round(100*np.max(M_S)/N_S)}%')
    ax[1].set_xlabel('Time [days]')
    ax[1].set_ylabel('Population')
    ax[1].set_ylim(0, N_S)
    ax[1].set_title(f'Disease spread in Scania')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    fig.canvas.draw_idle()  

fig, ax = plt.subplots(1, 2, figsize=(10, 6)) 
plt.subplots_adjust(bottom=0.25)

g_init = 0
plot_SIRD(g_init)

ax_g = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
g_slider = Slider(ax=ax_g, label='Qurantine rate ($\\gamma$)', valmin=0.0, valmax=0.15, valinit=g_init, valstep=0.001)

g_slider.on_changed(plot_SIRD)

plt.show()