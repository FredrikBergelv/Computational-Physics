# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:35:56 2024

@author: Fredrik Bergelv
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from matplotlib.animation import FuncAnimation, FFMpegWriter
from collections import Counter
import time
from numba import njit    

def initialize_grid(size, number):
    "This function initializes the cow field"
    
    grid = np.zeros((size, size))  # We create a grid
    grid_flatt = grid.flatten()    # We flatten the grid to make it simpler

    cow_indices = rd.sample(range(len(grid_flatt)), number)  # Randomize positions of the cows

    for i, index in enumerate(cow_indices):
        grid_flatt[index] = 1  # We give all the cows a unique odd number

    moving_cow = rd.choice(cow_indices)

    return grid_flatt, moving_cow  # We return the flattened grid and one random cow


def move_func(moving_cow, grid_flatt, size):
    "This function makes the cow move in a random way"

    paths = []
    # Handle boundaries to prevent out-of-bounds moves
    left = moving_cow - 1 if not moving_cow % size == 0 else None
    right = moving_cow + 1 if not moving_cow % size == size - 1 else None
    down = moving_cow - size if moving_cow >= size else None
    up = moving_cow + size if moving_cow < len(grid_flatt) - size else None

    # Check if neighboring pixels are occupied
    if left is not None and grid_flatt[left] == 0:  # Check left
        paths.append(1)  # Chance of moving left
    if right is not None and grid_flatt[right] == 0:  # Check right
        paths.append(2)  # Chance of moving right
    if down is not None and grid_flatt[down] == 0:  # Check down
        paths.append(3)  # Chance of moving down
    if up is not None and grid_flatt[up] == 0:  # Check up
        paths.append(4)  # Chance of moving up

    # If there are no valid paths, the cow stays in place
    if not paths:
        return moving_cow, grid_flatt

    walk_direction = rd.choice(paths)  # Randomize direction

    # Move left
    if walk_direction == 1:
        grid_flatt[left] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow = left  # Move the index to the new position

    # Move right
    elif walk_direction == 2:
        grid_flatt[right] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow = right  # Move the index to the new position

    # Move down
    elif walk_direction == 3:
        grid_flatt[down] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow = down  # Move the index to the new position

    # Move up
    elif walk_direction == 4:
        grid_flatt[up] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow = up  # Move the index to the new position

    return moving_cow, grid_flatt



def Imshow_animate(size, number, steps, style='video'):
    """
    This function runs the cow simulation and generates the animated imshow plot.
    """
    start_time = time.time()

    # Initialize everything
    grid_flatt, moving_cow = initialize_grid(size, number-1)
    frames = np.zeros((steps, size, size))  # We want to store all the grids

    # Track cow positions over time
    for i in range(steps):
        moving_cow, grid_flatt = move_func(moving_cow, grid_flatt, size)  # Make a cow step

        if grid_flatt[moving_cow] % 2 == 0:  # Since all cows are odd, if position is even: kill
            grid_flatt[moving_cow] = 0  # Kill cow
            if np.all(grid_flatt == 0):  # Check if all cows are gone
                print(f"Step {i}: All cows are dead, stopping simulation.")
                steps = i+1  # Update the number of frames
                break
        
        moving_cow = rd.choice(np.where(grid_flatt != 0)[0])  # Find new random cow

        grid = grid_flatt.reshape((size, size))
        frames[i] = grid  # Update the frame
        if i ==steps-1:
            print(f"Step {i}: The cows did not go extinct!")

    # Create the imshow animation
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(frames[0], cmap='jet')  # Use the first frame as the initial image

    # Set axis properties (similar to the other code)
    ax.set_xticks(np.arange(0, size, 1))  
    ax.set_yticks(np.arange(0, size, 1))
    ax.set_xticks(np.arange(0.5, size + 0.5, 1), minor=True)  
    ax.set_yticks(np.arange(0.5, size + 0.5), minor=True)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Killer Cow Simulation at step 0")

    def update(frame):
        if 100*frame % steps==0: # We print each time we do one percent 
            print(f'{round(100*frame/steps)}%')
            
        im.set_array(frames[frame])  # Update the image with the current frame
        ax.set_title(f"Killer Cow Simulation at step {frame}")  # Update the title with the current step
        return [im]

    # Create and run the animation using the pre-generated frames array
    animation = FuncAnimation(fig, update, frames=steps, blit=True, repeat=False)

    # Optionally, save the animation as a GIF or video
    if style=='video':
        animation.save(f'Imshow(size={size},number={number},steps={steps}).mp4', writer=FFMpegWriter(fps=60))
    if style=='gif':
        animation.save(f'Imshow(size={size},number={number},steps={steps}).gif', writer='pillow', fps=60)
        
    end_time = time.time()
    print(f"Elapsed time: {round(end_time - start_time)} seconds")
    
    
def Sactter_animate(size, number, steps, style='video'):
    """
    This function runs the cow scatter simulation and generates the animated scatter plot.
    """
    
    start_time = time.time()
    
    # Initialize everything
    grid_flatt, moving_cow = initialize_grid(size, number-1)
    grid = grid_flatt.reshape((size, size))  
    cow_positions = []  # Store cow positions for each step
    
    infected_grid_flatt, infected_moving_cow = initialize_grid(size, 1)
    infected_grid = grid_flatt.reshape((size, size))  
    infected_cow_positions = []  # Store cow positions for each step

    # Track cow positions over time
    for i in range(steps):
        moving_cow, grid_flatt = move_func(moving_cow, grid_flatt, size)  # Make a cow step
        infected_moving_cow, infected_grid_flatt = move_func(infected_moving_cow, infected_grid_flatt, size)  # Make a cow step


        if (grid_flatt[moving_cow] + infected_grid_flatt[infected_moving_cow]) == 2:  # Since all cows are odd, if position is even: kill
            grid_flatt[moving_cow] = 0 # Kill cows
            infected_grid_flatt[moving_cow] = 1 #replace with infected cow
            if np.all(grid_flatt == 0):  # Check if all cows are gone
                print(f"Step {i}: All cows are infected, stopping simulation.")
                steps = i+1
                break
            
        moving_cow = rd.choice(np.where(grid_flatt != 0)[0])  # Find new random cow
        infected_moving_cow = rd.choice(np.where(infected_grid_flatt != 0)[0])  # Find new random cow

            
        # Extract cow positions (x, y), do this by finding coordinates in flatt, and convert to normal grid
        current_cow_positions = np.array(np.unravel_index(np.where(grid_flatt != 0)[0], grid.shape)).T
        cow_positions.append(current_cow_positions)
        infected_current_cow_positions = np.array(np.unravel_index(np.where(infected_grid_flatt != 0)[0], grid.shape)).T
        infected_cow_positions.append(infected_current_cow_positions)
        if i ==steps-1:
            print(f"Step {i}: The cows did not go extinct!")

    last_frame = np.array([])  # Add an empty frame at the end
    cow_positions.append(last_frame)
    infected_cow_positions.append(last_frame)

    # Create the scatter plot animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, size-1)  
    ax.set_ylim(-0.5, size-1) 
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Cow Simulation at step 0")
    ax.set_xticks(np.arange(0, size, 1))  
    ax.set_yticks(np.arange(0, size, 1))  
    ax.set_xticks(np.arange(0.5, size+0.5, 1), minor=True)  
    ax.set_yticks(np.arange(0.5, size+0.5), minor=True)  
    ax.grid(True, which='minor', color='black', linewidth=0.5)

    # Function to update the scatter plot
    def update(frame):
        
        if 100*frame % steps==0: # We print each time we do one percent 
            print(f'{round(100*frame/steps)}%')
            
        ax.clear()  # Clear the previous plot
        ax.set_xlim(-0.5, size-1)  
        ax.set_ylim(-0.5, size-1) 
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_xticks(np.arange(0, size, 1))  
        ax.set_yticks(np.arange(0, size, 1))  
        ax.set_xticks(np.arange(0.5, size+0.5, 1), minor=True)  
        ax.set_yticks(np.arange(0.5, size+0.5), minor=True)  
        ax.grid(True, which='minor', color='black', linewidth=0.5)
        
        # Get cow positions for the current frame if we have cows
        if cow_positions[frame].size > 0:  # If there are still cows
            x, y = cow_positions[frame][:, 1], cow_positions[frame][:, 0]
            ax.scatter(x, y, c='blue', label='Cows', s=3.8e4/size**2, marker='o')  # Plot x, y positions
            
            infected_x, infected_y = infected_cow_positions[frame][:, 1], infected_cow_positions[frame][:, 0]
            ax.scatter(infected_x, infected_y, c='red', label='Infected Cows', s=3.8e4/size**2, marker='o')  # Plot x, y positions
            
            percent = 100*len(infected_cow_positions[frame])/(len(cow_positions[frame])+len(infected_cow_positions[frame]))
            
            ax.set_title(f"Cow Simulation at step {frame:.0f} with {percent:.0f}% infected")
            
            ax.legend(loc='upper right', markerscale=1)

    # Create and run the animation using the pre-generated cow positions
    animation = FuncAnimation(fig, update, frames=steps, blit=False, repeat=False)

    # Optionally, save the animation as a GIF or video
    if style=='video':
        animation.save(f'Infected_Sactter(size={size},number={number},steps={steps}).mp4', writer=FFMpegWriter(fps=30))
    if style=='gif':
        animation.save(f'Infected_Sactter(size={size},number={number},steps={steps}).gif', writer='pillow', fps=30)
        
    end_time = time.time()
    print(f"Elapsed time: {round(end_time - start_time)} seconds")


Sactter_animate(size=20, number=200, steps=300, style='video')

#Imshow_animate(size=5, number=20, steps=2000, style='video')


def Statistics_plot(size, number, steps, sample):
    
    def counter(size=size, number=number, steps=steps, sample=sample):
        """
        This function runs the cow simulation for many different times.
        """
    
        iteration_list = []  # We want to store all the values
        for k in range(sample):
            if 100*k % sample==0: # We print each time we do one percent 
                print(f'{round(100*k/sample)}%')
                
            # Initialize everything
            grid_flatt, moving_cow = initialize_grid(size, number-1)
            
            for i in range(steps):
                moving_cow, grid_flatt = move_func(moving_cow, grid_flatt, size)  # Make a cow step
        
                if grid_flatt[moving_cow] % 2 == 0:  # Since all cows are odd, if position is even: kill
                    grid_flatt[moving_cow] = 0  # Kill cow
                    if np.all(grid_flatt == 0):  # Check if all cows are gone
                        iteration_list.append(i)  # Update the number of frames
                        break
                    
                moving_cow = rd.choice(np.where(grid_flatt != 0)[0])  # Find new random cow
    
        return iteration_list
    
    
    iteration_list = (counter(size=size, number=number, steps=steps, sample=sample))
    
    iteration_list_count = Counter(iteration_list)
    
    # Sort the iteration_list by unique values 
    unique_values = sorted(iteration_list_count.keys())
    frequencies = [iteration_list_count[val] for val in unique_values]
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(unique_values, frequencies, color='red')
    
    # Add labels and title
    plt.xlabel('Iterations until convergence')
    plt.ylabel('Frequency')
    plt.title('Iterations until all cows are dead')
    
    # Display the chart
    plt.show()
    
#Statistics_plot(size=100, number=10, steps=10000, sample=10000)

def Convergence_plot(size, number, steps):
    
    grid_flatt, moving_cow = initialize_grid(size, number-1)
    
    density_list = []  # We want to store all the values
    
    for i in range(steps):
        moving_cow, grid_flatt = move_func(moving_cow, grid_flatt, size)  # Make a cow step

        if grid_flatt[moving_cow] % 2 == 0:  # Since all cows are odd, if position is even: kill
            grid_flatt[moving_cow] = 0  # Kill cow
            if np.all(grid_flatt == 0):  # Check if all cows are gone
                break
            
        moving_cow = rd.choice(np.where(grid_flatt != 0)[0])  # Find new random cow
        
        density = (len(np.where(grid_flatt != 0)[0])-1)/size**2
        density_list.append(density)
    
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(density_list))), density_list, label='Cow density')
    plt.grid(True)
    plt.xlabel('Steps')
    plt.ylabel('Cow density')
    plt.title('Plot of cow density per step')
    plt.legend()
    plt.show()
    
    
#Convergence_plot(size=20, number=300, steps=5000)
