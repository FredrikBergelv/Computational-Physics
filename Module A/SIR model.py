from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

b = np.mean([2.8,3.5])/np.mean([1,7])   # Rate of infection 
m = np.mean([30,75])/100/np.mean([1,7]) # Rate of mortality
g = 0.#02                               # Rate of vaccination 
a = 1/np.mean([1,7])-m                  # Rate of recovery

N = 300000
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

plt.figure(figsize=(15*0.6/2, 6*0.6))
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

plt.savefig(r'C:\Users\fredr\Downloads\Code\FYTN03\Module A\SIR.pdf')

