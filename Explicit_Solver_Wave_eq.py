import numpy as np
import matplotlib.pyplot as plt

L = 100

n = 25
dt = 0.5
dx = 0.5

lamb = .5

time = 0


def A(n):
    A = np.zeros((n, n))  
    for i in range(1, L - 1):
        A[i, i-1] = lamb**2
        A[i, i] = 2*(1-lamb**2)
        A[i, i+1] = lamb**2
    return A


def sites(L):
    sites = np.zeros(L)
    for k in range(L):
        if k>L/2 and k<L:
            sites[k] = np.sin(2*np.pi*k/L)
        else:
            sites[k] = 0
    return sites

A = A(L)
u = sites(L)

u_old = u
u_new = u

for i in range(n):
    u_temp = u_new
    u_new = A@u_new-u_old
    u_old = u_temp
    u_new[-1] = 0
    u_new[0] = 0


x = np.arange(L)  
plt.plot(x, u_new, marker='+', label='Explicit method')

plt.ylabel(r"$u(x)$")
plt.xlabel(r"$x$")
plt.grid()
plt.legend()
plt.show()