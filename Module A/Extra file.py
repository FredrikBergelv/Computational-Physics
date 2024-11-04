"""
Created on Mon Sep 10 2024
@author: Fredrik Bergelv
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

size = 0.6

b = np.mean([2.8,3.5])/np.mean([1,7])   # Rate of infection 
m = np.mean([30,75])/100/np.mean([1,7]) # Rate of mortality
g = 0.#01*2                               # Rate of vaccination 
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

InitialCond = I0_L, S0_L, R0_L, M0_L, I0_S, S0_S, R0_S, M0_S

def SIS(t, f): 
    N=150000
    I, S = f
    dIdt = b*S*I/N-a*I
    dSdt = -b*S*I/N+a*I
    return dIdt, dSdt

def SIR(t, f): 
    N=150000
    I, S, R = f
    dIdt = b*S*I/N-a*I
    dSdt = -b*S*I/N-g*S
    dRdt = g*S+a*I   
    return dIdt, dSdt, dRdt

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

sol = solve_ivp(SIRD_travel, [t0, t1], InitialCond, 
                method='RK45', t_eval=np.linspace(t0, t1, h))

for i in range(0,8):
    sol.y[i]=sol.y[i]/1000
I_L, S_L, R_L, M_L = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
I_S, S_S, R_S, M_S = sol.y[4], sol.y[5], sol.y[6], sol.y[7]
t = sol.t

plt.figure(figsize=(15*size, 6*size))

plt.subplot(1, 2, 1)
plt.plot(t, I_L, label='Infected')
plt.plot(t, S_L, label='Susceptible')
plt.plot(t, R_L, label='Recovered')
plt.plot(t, M_L, label=f'Deaths = {round(100*np.max(M_L*1e3)/N_L)}%')
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
plt.plot(t, M_S, label=f'Deaths = {round(100*np.max(M_S*1e3)/N_S)}%')
plt.xlabel('Time [days]')
plt.ylabel('Population [thousands]')
plt.title(f'Disease spread in Scania with $\gamma={int(g*100)}\%$')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()



"""
plt.figure()
TotPop = sol.y[0] + sol.y[1] + sol.y[2] + sol.y[4] + sol.y[5] + sol.y[6]
plt.plot(t, TotPop, label='Total population', c='black')
plt.xlabel('Time [days]')
plt.ylabel('Population')
plt.title('Total deaths')
plt.ylim(0,N_L+N_S+10000)
plt.legend()
plt.grid(True)  """


sol_h = solve_ivp(SIRD_travel, [t0, t1], InitialCond, 
                method='RK45', t_eval=np.linspace(t0, t1, h))
sol_2h = solve_ivp(SIRD_travel, [t0, t1], InitialCond, 
                method='RK45', t_eval=np.linspace(t0, t1, h*2))

Titles = ('Infected in Lübeck', 'Susceptible in Lübeck', 
          'Recovered in Lübeck', 'Deaths in Lübeck',  
         'Infected in Scania', 'Susceptible in Scania', 
         'Recovered in Scania', 'Deaths in Scania')

m = 4
plt.figure(figsize=(7*0.8/0.5*size, 5*0.8/0.5*size))
for i in range(0,8):
    Err = (sol_h.y[i]-np.interp(sol_h.t,sol_2h.t,sol_2h.y[i]))/(2**m-1)
    plt.plot(sol_h.t, Err, label=Titles[i])
plt.xlabel('Time [days]')
plt.ylabel('Absolute error')
plt.title('Absolute error in disease spread')
plt.legend()
plt.grid(True)
plt.legend(loc='lower right')

plt.tight_layout()


############################################################################


from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

b = np.mean([2.8,3.5])/np.mean([1,7])   # Rate of infection 
m = np.mean([30,75])/100/np.mean([1,7]) # Rate of mortality
g = 0.02                               # Rate of vaccination 
a = 1/np.mean([1,7])-m                  # Rate of recovery

N = 150000
I0 = 13
timescale = t0, t1 = 0, 100
InitalConditions = (I0, N-I0, 0)

def SIR(t, f): 
    I, S, R = f
    dIdt = b*S*I/N-a*I
    dSdt = -b*S*I/N-g*S
    dRdt = g*S+a*I   
    return dIdt, dSdt, dRdt


sol = solve_ivp(SIR, [t0, t1], InitalConditions, method='RK45', 
                     t_eval=np.linspace(t0, t1, 1000))

I = sol.y[0]/1000
S = sol.y[1]/1000
R = sol.y[2]/1000
t = sol.t

plt.figure(figsize=(15*size/2, 6*size))
plt.plot(t, I, label='Infected')
plt.plot(t, S, label='Susceptible')
plt.plot(t, R, label='Recovered')


plt.xlabel('Time [days]')
plt.ylabel('Population [thousands]')
plt.title(f'Infection spread with $\gamma={int(g*100)}\%$')
plt.legend()
plt.grid(True)
plt.show()

plt.tight_layout()



from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

b = np.mean([2.8,3.5])/np.mean([1,7])   # Rate of infection 
m = np.mean([30,75])/100/np.mean([1,7]) # Rate of mortality
g = 0#.05                               # Rate of vaccination 
a = 1/np.mean([1,7])-m                  # Rate of recovery

N = 150000
I0 = 13
timescale = t0, t1 = 0, 40
InitalConditions = (I0, N-I0)

def SIS(t, f): 
    I, S = f
    dIdt = b*S*I/N-a*I
    dSdt = -b*S*I/N+a*I
    return dIdt, dSdt


sol = solve_ivp(SIS, [t0, t1],InitalConditions , method='RK45', 
                     t_eval=np.linspace(t0, t1, 1000))

I = sol.y[0]/1000
S = sol.y[1]/1000
t = sol.t

plt.figure(figsize=(15*size/2, 6*size))
plt.plot(t, I, label='Infected')
plt.plot(t, S, label='Susceptible')
plt.plot(t, N*(1-a/b)/1000+t*0, label=r'$\left(1-\frac{\alpha}{\beta}\right)N$',
         c='black', linestyle='dashed')


plt.xlabel('Time [days]')
plt.ylabel('Population [thousands]')
plt.title(f'Infection spread in the population')
plt.legend()
plt.grid(True)

plt.tight_layout()


#############################

#plt.close('all')