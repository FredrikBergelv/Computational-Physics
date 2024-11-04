import numpy as np
import matplotlib.pyplot as plt



h = 0.25

A = np.array([
    [-2 - h**2, 1, 0],
    [1, -2 - h**2, 1],
    [0, 1, -2 - h**2]])


b = np.array([0, 0, -np.sinh(1)])

y = np.linalg.solve(A, b)

y_sol = np.array([0, y[0], y[1], y[2], np.sinh(1)])

print(y_sol)

x = np.array([0, 0.25, 0.5, 0.75, 1])


for i in range(0,5):
    Err = (y_sol[i]-np.sinh(x[i]))/np.sinh(x[i])
    print(Err*1e4)
    
plt.scatter(x,y_sol, label='Numerical solution')
plt.plot(x,np.sinh(x), c='b', label=r'$y(x)=\sinh(x)$')
plt.legend()

