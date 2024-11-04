from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

b = np.mean([2.8,3.5])/np.mean([1,7])   # Rate of infection 
m = np.mean([30,75])/100/np.mean([1,7]) # Rate of mortality
g = 0#.05                               # Rate of vaccination 
a = 1/np.mean([1,7])-m                  # Rate of recovery

N = 300000
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

plt.figure(figsize=(15*0.5/2, 6*0.5))
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

plt.savefig(r'C:\Users\fredr\Downloads\Code\FYTN03\Module A\SIS.pdf')
