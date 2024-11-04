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
size = 5
number = 10
steps = 100


grid_flatt, moving_cow = initialize_grid(size, number-1)
grid = grid_flatt.reshape((size, size))  


frames = np.zeros((steps, size,size)) # We want to store all the grids 

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
        
    grid = grid_flatt.reshape((size, size)) 
    frames[i] = grid # Update the frame

fig, ax = plt.subplots()
im = ax.imshow(frames[0], cmap='inferno')  # Use the first frame as the initial image
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_xticks(np.arange(0, size, 1))  
ax.set_yticks(np.arange(0, size, 1))
ax.set_xticks(np.arange(0.5, size + 0.5, 1), minor=True)  
ax.set_yticks(np.arange(0.5, size + 0.5), minor=True)
ax.set_title("Killer Cow Simulation")


def update(frame):
    im.set_array(frames[frame])  # Update the image with the current frame
    ax.set_title(f"Killer Cow Simulation at step {frame}")  # Update the title with the current step

    return [im]

# Create and run the animation using the pre-generated frames array
ani = FuncAnimation(fig, update, frames=steps, blit=True, repeat=False)
plt.show()

ani.save('cow_simulation.gif', writer='pillow', fps=60)  # You can adjust the fps (frames per second)

