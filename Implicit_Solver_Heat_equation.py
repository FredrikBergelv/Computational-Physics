import numpy as np
import matplotlib.pyplot as plt

N = 10
T0 = 0
T1 = 2
dt = 0.5
k = 1
h = 0.97
time = 50


def A(n,k,dt,h):
    f = k * dt / (h**2)
    A = np.zeros((n, n))  
    for i in range(1, N - 1):
        A[i, i-1] = f
        A[i, i] = -2 *f
        A[i, i+1] = f
    return A


def sites(N, T0, T1):
    sites = np.zeros(N)
    sites[0] = T0
    sites[-1] = T1
    return sites

def sol_explicit(T):
    return (I + A) @ T

def sol_implicit(T):
    return np.linalg.solve((I - A),T)

I = np.eye(N)
A = A(N,k,dt,h)
T = sites(N, T0, T1)

T_explicit = T
T_imblicit = T
for i in range(time):
    T_explicit = sol_implicit(T_explicit)
    T_imblicit = sol_explicit(T_imblicit)

x = np.arange(N)  

plt.plot(x, T_imblicit, marker='+', label='Explicit method')
plt.plot(x, T_explicit, marker='+', label='Implicit method')
plt.ylim(T0,T1)
plt.ylabel(r"$T(x)$")
plt.xlabel(r"$x$")
plt.title(f"Solved with h={h}")
plt.grid()
plt.legend()
plt.show()