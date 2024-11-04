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
from scipy.optimize import curve_fit

  

def initialize_grid(size, number):
    "This function initializes the cow field"
    
    grid = np.zeros((size, size))  # We create a grid
    grid_flatt = grid.flatten()    # We flatten the grid to make it simpler

    cow_indices = rd.sample(range(len(grid_flatt)), number+1)  # Randomize positions of the cows

    for i, index in enumerate(cow_indices):
        grid_flatt[index] = 1  # We give all the cows a unique odd number

    moving_cow = rd.choice(cow_indices)

    return grid_flatt, moving_cow  # We return the flattened grid and one random cow


def move_func(moving_cow, grid_flatt, size):
    "This function makes the cow move in a random way"

    paths = []
    # Handle boundaries to prevent out-of-bounds moves
    if not moving_cow % size == 0:          # Left boundary
        paths.append(1)                     # Chance of moving left
    if not moving_cow % size == size - 1:   # Right boundary
        paths.append(2)                     # Chance of moving right
    if moving_cow >= size:                  # Bottom boundary
        paths.append(3)                     # Chance of moving down
    if moving_cow < len(grid_flatt) - size: # Top boundary
        paths.append(4)                     # Chance of moving up

    walk_direction = rd.choice(paths)  # Randomize direction

    # Move left
    if walk_direction == 1:
        grid_flatt[moving_cow - 1] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow -= 1  # Move the index to the new position

    # Move right
    if walk_direction == 2:
        grid_flatt[moving_cow + 1] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow += 1  # Move the index to the new position

    # Move down
    if walk_direction == 3:
        grid_flatt[moving_cow - size] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow -= size  # Move the index to the new position

    # Move up
    if walk_direction == 4:
        grid_flatt[moving_cow + size] += grid_flatt[moving_cow]
        grid_flatt[moving_cow] = 0
        moving_cow += size  # Move the index to the new position

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
        # Make a cow step
        moving_cow, grid_flatt = move_func(moving_cow, grid_flatt, size)  

        if grid_flatt[moving_cow] % 2 == 0:  # if vlaue is 2: kill
            grid_flatt[moving_cow] = 0   # Kill cow
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
        ax.set_title(f"Killer Cow Simulation at step {frame}")  # Update the title 
        return [im]

    # Create and run the animation using the pre-generated frames array
    animation = FuncAnimation(fig, update, frames=steps, blit=True, repeat=False)

    # Optionally, save the animation as a GIF or video
    if style=='video':
        animation.save(f'Imshow(size={size},number={number},steps={steps}).mp4', 
                       writer=FFMpegWriter(fps=60))
    if style=='gif':
        animation.save(f'Imshow(size={size},number={number},steps={steps}).gif', 
                       writer='pillow', fps=60)
        
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

    # Track cow positions over time
    for i in range(steps):
        # MAke a cow step
        moving_cow, grid_flatt = move_func(moving_cow, grid_flatt, size)  

        if grid_flatt[moving_cow] % 2 == 0:  # If value is 2: kill
            grid_flatt[moving_cow] = 0  # Kill cows
            if np.all(grid_flatt == 0):  # Check if all cows are gone
                print(f"Step {i}: All cows are dead, stopping simulation.")
                steps = i+1
                break
            
        moving_cow = rd.choice(np.where(grid_flatt != 0)[0])  # Find new random cow
            
        # Extract cow positions (x,y), by finding coordinates in flatt and thus normal grid
        current_cow_positions = np.array(np.unravel_index(np.where(grid_flatt != 0)[0], 
                                                          grid.shape)).T
        cow_positions.append(current_cow_positions)
        if i ==steps-1:
            print(f"Step {i}: The cows did not go extinct!")

    last_frame = np.array([])  # Add an empty frame at the end
    cow_positions.append(last_frame)

    # Create the scatter plot animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, size-1)  
    ax.set_ylim(-0.5, size-1) 
    ax.set_xlabel("Width")
    ax.set_ylabel("Depth")
    ax.set_title("Cow Simulation at Step 0")
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
        ax.set_xlabel("Width [pixels]")
        ax.set_ylabel("Depth [pixels]")
        ax.set_title(f"Cow Simulation at Step {frame:.0f}")
        ax.set_xticks(np.arange(0, size, 1))  
        ax.set_yticks(np.arange(0, size, 1))  
        ax.set_xticks(np.arange(0.5, size+0.5, 1), minor=True)  
        ax.set_yticks(np.arange(0.5, size+0.5), minor=True)  
        ax.grid(True, which='minor', color='black', linewidth=0.5)
        
        # Get cow positions for the current frame if we have cows
        if cow_positions[frame].size > 0:  # If there are still cows
            x, y = cow_positions[frame][:, 1], cow_positions[frame][:, 0]
            ax.scatter(x, y, c='red', label='Cows', s=3.8e4/size**2, marker='o')  
            ax.legend(loc='upper right', markerscale=1)

    # Create and run the animation using the pre-generated cow positions
    animation = FuncAnimation(fig, update, frames=steps, blit=False, repeat=False)

    # Optionally, save the animation as a GIF or video
    if style=='video':
        animation.save(f'Sactter(size={size},number={number},steps={steps}).mp4', 
                       writer=FFMpegWriter(fps=60))
    if style=='gif':
        animation.save(f'Sactter(size={size},number={number},steps={steps}).gif', 
                       writer='pillow', fps=60)
        
    end_time = time.time()
    print(f"Elapsed time: {round(end_time - start_time)} seconds")


def Statistics_plot(size, number, steps, sample, save=False):
    """ This function displays how many iterations we need to kill all cows"""
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
                # Make a cow step
                moving_cow, grid_flatt = move_func(moving_cow, grid_flatt, size)  
        
                if grid_flatt[moving_cow] % 2 == 0:  # If value is 2: Kill
                    grid_flatt[moving_cow] = 0  # Kill cow
                    if np.all(grid_flatt == 0):  # Check if all cows are gone
                        iteration_list.append(i)  # Update the number of frames
                        break
                    
                moving_cow = rd.choice(np.where(grid_flatt != 0)[0])  # Find new random cow
    
        return iteration_list
    
    # Count all the iterations we gathered
    iteration_list = (counter(size=size, number=number, steps=steps, sample=sample))
    iteration_list_count = Counter(iteration_list)
    
    # Sort the iteration_list by unique values 
    unique_values = sorted(iteration_list_count.keys())
    frequencies = np.array([iteration_list_count[val] for val in unique_values])
    frequencies = frequencies / np.sum(frequencies)
    
    # Create the bar chart
    plt.figure(figsize=(6, 5))
    plt.bar(unique_values, frequencies, color='red')
    
    # Add labels and title
    plt.xlabel('Iterations until convergence')
    plt.ylabel('Normalized frequency')
    plt.title('Iterations Until All Cows Are Dead')
    
    # Display the chart
    plt.show()
    
    # Optionally save 
    if save == True :
        plt.savefig(f'Statistics(depth={size},number={number},sample={sample}).pdf')
    

def Convergence_plot(size, number, steps, save=False):
    
    sample = 1000  # Number of iterations for averaging
    
    density_list = np.zeros((steps, sample))  # Store density values for all iterations
    
    for k in range(sample):
        # Initialize the grid and cows
        grid_flatt, moving_cow = initialize_grid(size, number-1)
    
        for i in range(steps):
            # Make a cow step
            moving_cow, grid_flatt = move_func(moving_cow, grid_flatt, size)  
    
            if grid_flatt[moving_cow] % 2 == 0:  # If value is 2: kill cow
                grid_flatt[moving_cow] = 0  # Kill cow
                if np.all(grid_flatt == 0):  # Check if all cows are gone
                    break
                
            # Find a new random cow
            moving_cow = rd.choice(np.where(grid_flatt != 0)[0])  
            
            # Calculate the density of cows
            density = (len(np.where(grid_flatt != 0)[0]) - 1) / size**2
            density_list[i, k] = density  # Store density for this iteration
            
    # Calculate the mean density over all iterations
    mean_density = np.mean(density_list, axis=1)

    # Define the fitting function: A / x^n
    def f(x, A, n):
        return A / (x**n)

    # Prepare the data for fitting
    x = np.arange(1, len(mean_density) + 1)  # Steps from 1 to number of steps
    half_index = len(x) // 10  # Find from where we want to fit the data
    last_half_x = x[half_index:]  
    last_half_density = mean_density[half_index:]  # Take the last half of the density values

    # Perform curve fitting on the last half
    fit, _ = curve_fit(f, last_half_x, last_half_density)

    # Generate values for the fit line
    x_val = np.linspace(0, max(last_half_x), 1000)

    # Plot the results
    plt.plot(x_val, f(x_val, *fit), c='r', 
             label=rf'$\rho_{{cow}} = {fit[0]:.2f}/steps^{{{fit[1]:.2f}}}$')
    plt.ylim(0,0.5)
    plt.plot(x, mean_density, label='Simulated Data')
    plt.grid(True)
    plt.xlabel('Steps')
    plt.ylabel(r'$\rho_{{cow}}$ [cows/pixels$^2$]')
    plt.title('Plot of Cow Density per Step')
    plt.legend()
    plt.show()

    # Save the plot if needed
    if save:
        plt.savefig(f'Convergence(size={size},number={number},steps={steps}).pdf')
       
    
def Many_convergence_plot(size, number_list, steps, save=False):
    """ This shows the convergence for different densities"""
    # Plot the result in the same plot
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    plt.xlabel('Steps')
    plt.ylabel(r'$\rho_{{cow}}$ [cows/pixels$^2$]')
    plt.title('Plot of Cow Density per Step')
    for number in number_list:
        grid_flatt, moving_cow = initialize_grid(size, number-1)
        
        density_list = []  # We want to store all the values
        
        for i in range(steps):
            # Make a cow step
            moving_cow, grid_flatt = move_func(moving_cow, grid_flatt, size) 
    
            if grid_flatt[moving_cow] % 2 == 0:  # If value is 2: kill
                grid_flatt[moving_cow] = 0  # Kill cow
                if np.all(grid_flatt == 0):  # Check if all cows are gone
                    break
                
            moving_cow = rd.choice(np.where(grid_flatt != 0)[0])  # Find new cow
            
            density = (len(np.where(grid_flatt != 0)[0])-1)/size**2
            density_list.append(density) # Append the result
        # Plot the result
        plt.plot(list(range(len(density_list))), density_list, 
                 label=rf'$\rho_{{cows}}^{{0}}=${number/(size**2):.2f}')

    plt.legend()
    plt.show()
    
    if save == True :
        plt.savefig(f'Convergence2(size={size},number={number_list},steps={steps}).pdf')

      
    
#Sactter_animate(size=20, number=200, steps=5000, style='video')

Convergence_plot(size=30, number=200, steps=5000, save=True)

#Statistics_plot(size=20, number=200, steps=15000, sample=100000, save=True)

#Many_convergence_plot(size=20, number_list=[100, 200, 300, 400], steps=3000, save=True)

