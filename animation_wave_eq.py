import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        if k >= L/2 and k <= L:
            sites[k] = -np.sin(2*np.pi*k/L)
        else:
            sites[k] = 0
    return sites

A = A(L)
u = sites(L)
u_old = np.copy(u)
u_new = np.copy(u)

fig, ax = plt.subplots()
x = np.arange(L)
plt.ylabel(r"$u(x)$")
plt.xlabel(r"$x$")
plt.title('Animation of wave equation')
plt.grid()
plt.ylim(-1,1)


def update(frame):
    global u_new, u_old, time
    line, = ax.plot(x, u_new, c='r', linewidth=3)
    u_temp = np.copy(u_new)
    u_new = A @ u_new - u_old
    u_old = np.copy(u_temp)
    u_new[-1] = 0
    u_new[0] = 0   

    line.set_ydata(u_new)
    return line,

ani = animation.FuncAnimation(fig, update, frames=n, blit=True, interval=5)

plt.show()
