import numpy as np
import matplotlib.pyplot as plt

# Initial conditions and settings
y0 = np.array([[1], [1], [1]])
A = np.array([[-4, 1, 0],
              [1, -4, 1],
              [0, 1, -4]])


eig = eig1, eig2, eig3 = np.array([-4-np.sqrt(2), -4,  -4+np.sqrt(2)])

h_values = [2/max(np.abs(eig))+0.01]



# Plot for each h value
plt.figure(figsize=(12, 8))
for h in h_values:
    y = y0  # Reset initial condition for each h
    results = [y.flatten()]  # Store results for each time step
    
    for _ in range(100):
        y = y + h * A @ y
        results.append(y.flatten())
        
    results = np.array(results)  # Convert to array for easier plotting
    
    # Plot each component of y for the given h
    plt.plot(results, label=f'$y_1$ (h={h:.3f})')


# Labeling
plt.xlabel("Time step")
plt.ylabel("Values of $y$ components")
plt.legend()
plt.title("Explicit Euler Integration for Different Step Sizes h")
plt.grid()
plt.show()