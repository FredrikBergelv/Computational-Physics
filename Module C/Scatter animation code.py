# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:35:56 2024

@author: Fredrik Bergelv
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd 
from matplotlib.animation import FuncAnimation

def initialize_grid(size, number):
    "This function initialize the cow field"
    
    grid  = np.zeros((size,size)) # We create a grid
    grid_flatt = grid.flatten()   # We flatten the grid to make it simplier
        
    cow_indices = rd.sample(range(len(grid_flatt)), number+1) # Randomize postions of the cows

    for i, index in enumerate(cow_indices):
        grid_flatt[index] = 2*i+1 # We give all the cows a unique odd number
        
    moving_cow = rd.choice(cow_indices)
    
    return grid_flatt, moving_cow #We return where the flatt grid and one random cow


def move_func(moving_cow, grid_flatt):
    "This functionmakes the cow move in a random way"
    
    paths = []
    # Here we look at the Dirichlet boundaries
    if not moving_cow % size == 0: # If index dvided by size is 0, then we are on the left wall and can't go left
        paths.append(1) # Chance of moving left added
    if not moving_cow % size == size - 1: # If index dvided by size is size-1, then we are on the right wall and can't go right
        paths.append(2) # Chance of moving right added             
    if moving_cow >= size: # If index is larger then size, then we are on the bottom row
        paths.append(3) # Chance of moving down added
    if moving_cow < len(grid_flatt) - size:  # Check if we are on the top row
        paths.append(4) # Chance of moving up added
    
    walk_direction = rd.choice(paths) # We randomize 1 or 2 or 3 or 4

    # Move left
    if walk_direction == 1:
        grid_flatt[moving_cow - 1] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow -= 1 #We move the index to the new position
 
    # Move right
    if walk_direction == 2:
        grid_flatt[moving_cow + 1] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow += 1 #We move the index to the new position

    # Move down
    if walk_direction == 3:
        grid_flatt[moving_cow - size] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow -= size #We move the index to the new position
    
    # Move up
    if walk_direction == 4:
        grid_flatt[moving_cow + size] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow += size #We move the index to the new position

    return moving_cow, grid_flatt

# Initialize everything
size = 10
number = 20
steps = 1000


grid_flatt, moving_cow = initialize_grid(size, number-1)
grid = grid_flatt.reshape((size, size))  

cow_positions = []  # Store cow positions for each step

# Track cow positions over time
for i in range(steps):
    "This loop changes each frame"
    
    moving_cow, grid_flatt = move_func(moving_cow, grid_flatt)  # Make a cow step
    
    if grid_flatt[moving_cow] % 2 == 0:  # Since all cows are odd, if position is even: kill
        grid_flatt[moving_cow] = 0  # Kill cows
        if np.all(grid_flatt == 0):  # Check if all cows are gone
            print(f"Step {i}: All cows are dead, stopping simulation.")
            steps = i+1
            break
        else:
            moving_cow = rd.choice(np.where(grid_flatt != 0)[0])  # Find new random cow
    
    # Extract cow positions (x, y), do this by finidng coordenates in flatt, and convert to normal grid
    current_cow_positions = np.array(np.unravel_index(np.where(grid_flatt != 0)[0], grid.shape)).T 
    cow_positions.append(current_cow_positions)

last_frame =  np.array([])
cow_positions.append(last_frame)


# Create the scatter plot animation
fig, ax = plt.subplots()
ax.set_xlim(-0.5, size-1)  
ax.set_ylim(-0.5, size-1) 
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Cow Simulation at step 0")
ax.set_xticks(np.arange(0, size, 1))  
ax.set_yticks(np.arange(0, size, 1))  
ax.set_xticks(np.arange(0.5, size+.5, 1), minor=True)  
ax.set_yticks(np.arange(0.5, size+.5), minor=True)  
ax.grid(True, which='minor', color='black', linewidth=0.5)


# Function to update the scatter plot
def update(frame):
    ax.clear()  # Clear the previous plot
    ax.set_xlim(-0.5, size-1)  
    ax.set_ylim(-0.5, size-1) 
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title(f"Cow Simulation at step {frame:.0f}")
    ax.set_xticks(np.arange(0, size, 1))  
    ax.set_yticks(np.arange(0, size, 1))  
    ax.set_xticks(np.arange(0.5, size+.5, 1), minor=True)  
    ax.set_yticks(np.arange(0.5, size+.5), minor=True)  
    ax.grid(True, which='minor', color='black', linewidth=0.5)
    
    # Get cow positions for the current frame if we have cows
    if cow_positions[frame].size > 0:  # If there are still cows
        x, y = cow_positions[frame][:, 1], cow_positions[frame][:, 0]
        ax.scatter(x, y, c='red', label='Cows', s=3.8e4/size**2, marker='o')  # Plot x, y positions
        ax.legend(loc='upper right', markerscale=0.1)
    

# Create and run the animation using the pre-generated cow positions
animation = FuncAnimation(fig, update, frames=steps, blit=False, repeat=False)
plt.show()

# Save the animation (optional)
animation.save('cow_scatter_simulation.gif', writer='pillow', fps=30)
