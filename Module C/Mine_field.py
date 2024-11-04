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
    """Initialize the minefield grid with no mines on the t.op and bottom row"""
    grid = np.zeros((size, size)) # Create an empty grid with area size*size
    mine_flatt = grid.flatten()   # Flatten the grid for easier indexing

    # Total cells excluding the first and last rows
    total_cells = (size - 2) * size

    # Ensure we don't request more mines than available cells
    if number > total_cells:
        raise ValueError("Number of mines exceeds available cells in the grid.")

    # Randomize positions of the mines excluding the first and last rows
    available_indices = list(range(size * 1, size * (size - 1)))  # Exclude first and last rows
    mine_indices = rd.sample(available_indices, number)  # Randomly sample mine positions

    for index in mine_indices:
        mine_flatt[index] = 1  # Place mines in the selected positions

    return mine_flatt

def move_func(moving_cow, grid_flatt, size):
    "This function makes the cow move one step in a random direction"
    
    paths = []
    # Handle boundaries to prevent out-of-bounds moves

    if not moving_cow % size == 0:          # Left boundary
        paths.append(1)                     # As chance of moving left
    if not moving_cow % size == size - 1:   # Right boundary
        paths.append(2)                     # Ad chance of moving right
    if moving_cow >= size:                  # Bottom boundary
        paths.append(3)                     # Ad chance of moving down
    if moving_cow < len(grid_flatt) - size: # Top boundary
        paths.append(4)                      
        paths.append(4)                     # The chance of moving up should be 50%
        paths.append(4)
        paths.append(4)

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

    return moving_cow, grid_flatt # Return the new moving cow and the new grid
    
def Sactter_animate(size, number, style='video'): 

    """This function shows that the cow deaths fit a Gaussian"""
    

    start_time = time.time()
    
    # Initialize everything
    grid = np.zeros((size, size))  
    mine_flatt = initialize_grid(size, number)
    grid_flatt = initialize_grid(size, 0) # We start with no cow

    cow_positions  = []  # Store cow positions for each step
    mine_positions = []  # Store mine positions for each step
    cow_counter    = []  # Store which cow iteration we are on
    
    frames = 0 # Count how many frames we want to display
    cow = 1 # Count which cow we are on
    
    done = False
    while done==False: # Iterate until one cow makes it across the field
        frames += 1 
        moving_cow = int(size/2) # The starting poition of the cow 
        grid_flatt[moving_cow] = 1 
        
        # Update all the lists where we store our values on the form (x,y)
        current_cow_positions = np.array(np.unravel_index(np.where(grid_flatt != 0)[0], 
                                                          grid.shape)).T
        cow_positions.append(current_cow_positions)
        cow_counter.append(cow)
        current_mine_positions = np.array(np.unravel_index(np.where(mine_flatt != 0)[0], 
                                                           grid.shape)).T
        mine_positions.append(current_mine_positions)

        for i in range(1000): # Loop through the movment of the cow
            frames += 1
        
            moving_cow, grid_flatt = move_func(moving_cow, grid_flatt, size)  # Make a cow step
            
            # Update all the lists where we store our values on the form (x,y)
            current_cow_positions = np.array(np.unravel_index(np.where(grid_flatt != 0)[0], 
                                                              grid.shape)).T
            cow_positions.append(current_cow_positions)
            cow_counter.append(cow)
            current_mine_positions = np.array(np.unravel_index(np.where(mine_flatt != 0)[0], 
                                                               grid.shape)).T
            mine_positions.append(current_mine_positions)
    
            # Since all cows are 1, if position is 2: kill
            if (grid_flatt[moving_cow]+mine_flatt[moving_cow]) == 2:  
                grid_flatt[moving_cow] = 0  # Kill cows
                mine_flatt[moving_cow] = 0  # Remove the mine
                cow += 1
            
                print(f"Step {i}: The cow stepped on a mine.")
                break
            
            # Check if we are on the other side
            if moving_cow > len(grid_flatt) - size: 
                print(f"Step {i}: The cow made it across the mine field.")
                done = True # Stop the while-loop
                break
        
                
    # Create the scatter plot animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, size-1)  
    ax.set_ylim(-0.5, size-1) 
    ax.set_xlabel(r"Width $\left[\text{pixels}\right]$")
    ax.set_ylabel(r"Depth $\left[\text{pixels}\right]$")
    ax.set_title("Cow Number 1 at Step 0")
    ax.set_xticks(np.arange(0, size, 1))  
    ax.set_yticks(np.arange(0, size, 1))  
    ax.set_xticks(np.arange(0.5, size+0.5, 1), minor=True)  
    ax.set_yticks(np.arange(0.5, size+0.5), minor=True)  
    ax.grid(True, which='minor', color='black', linewidth=0.5)

    # Function to update the scatter plot
    def update(frame):
        if 100*frame % frames==0: # We print each time we do one percent 
            print(f'{round(100*frame/frames)}%')
            
        ax.clear()  # Clear the previous plot
        ax.set_xlim(-0.5, size-1)  
        ax.set_ylim(-0.5, size-1) 
        ax.set_xlabel("Width [pixels]")
        ax.set_ylabel("Depth [pixels]")
        ax.set_title(f"Cow Number {cow_counter[frame]:.0f} at Step {frame:.0f}")
        ax.set_xticks(np.arange(0, size, 1))  
        ax.set_yticks(np.arange(0, size, 1))  
        ax.set_xticks(np.arange(0.5, size+0.5, 1), minor=True)  
        ax.set_yticks(np.arange(0.5, size+0.5), minor=True)  
        ax.grid(True, which='minor', color='black', linewidth=0.5)
        
        # Color the grid to make it easier to view
        grid_colors = np.ones((size, size, 3))  # Initialize grid with white (all ones)
        
        # Light green for top and bottom rows
        grid_colors[0, :, :] = [0.6, 1.0, 0.6]    # Green for bottom rows
        grid_colors[-1, :, :] = [0.6, 1.0, 0.6]   # Green for top rows
        grid_colors[1:-1, :, :] = [1.0, 0.6, 0.6] # Red for middle rows
        grid_colors[0, int(size/2), :] = [1.0, 1.0, 0.0] # Color of starting position
       
        # Plot grid as background
        ax.imshow(grid_colors, extent=[-0.5, size-0.5, -0.5, size-0.5], origin='lower')

        # Get cow positions for the current frame if we have cows
        if cow_positions[frame].size > 0:  # If there are still cows
            # Plot x, y positions for cow
            x, y = cow_positions[frame][:, 1], cow_positions[frame][:, 0]
            ax.scatter(x, y, c='blue', label='Cow', s=3.8e4/size**2, marker='o')  
            # Plot x, y positions for mines
            x_mine, y_mine = mine_positions[frame][:, 1], mine_positions[frame][:, 0]
            ax.scatter(x_mine, y_mine, c='darkred', label='Mines', s=3.8e4/size**2, marker='x')  
        
        ax.legend(loc='upper right', markerscale=1/2)
        

    # Create and run the animation using the pre-generated cow positions
    animation = FuncAnimation(fig, update, frames=frames, blit=False, repeat=False)

    # Optionally, save the animation as a GIF or video
    if style=='video':
        animation.save(f'Mine_field(size={size},number={number},steps={frames}).mp4', 
                       writer=FFMpegWriter(fps=2))
    if style=='gif':
        animation.save(f'Mine_field(size={size},number={number},steps={frames}).gif', 
                       writer='pillow', fps=10)
        
    end_time = time.time()
    print(f"Elapsed time: {round(end_time - start_time)} seconds")
    
def counter(size, number, sample):
    """ This function runs the cow simulation for many different times."""

    iteration_list = []  # We want to store all the values for how long it took to get across
    
    for k in range(sample): # We print each time we do one percent
        
        # Initialize everything
        grid_flatt = initialize_grid(size, 0)
        mine_flatt = initialize_grid(size, number)

        cow_counter = []
        
        frames = 0 
        cow = 1
        # Track cow positions over time
        done = False
        while done==False:
            frames += 1
            moving_cow = int(size/2)
            grid_flatt[moving_cow] = 1
            
            cow_counter.append(cow) 
            
            for i in range(1000):
                frames += 1
                # Make a cow step
                moving_cow, grid_flatt = move_func(moving_cow, grid_flatt, size)  
                
                # append the cow position
                cow_counter.append(cow)
                # If the cow steps on a mine kill everything
                if (grid_flatt[moving_cow] + mine_flatt[moving_cow]) == 2:  
                    grid_flatt[moving_cow] = 0  # Kill cows
                    mine_flatt[moving_cow] = 0  # Remove the mine
                    cow += 1
                    break
                # Check if the cow made it across the field 
                if moving_cow > len(grid_flatt) - size:
                    done = True
                    iteration_list.append(cow) # Add the number of iterations
                    break

    return iteration_list

def Statistics_plot(size, number, sample, save=False):
    """ This function shows how many iterations it takes to make it across"""
    
    iteration_list = counter(size=size, number=number, sample=sample)
    iteration_list_count = Counter(iteration_list) # Count all the iterations
    
    # Sort the iteration_list by unique values 
    unique_values = np.array(sorted(iteration_list_count.keys()))
    frequencies = np.array([iteration_list_count[val] for val in unique_values])
    frequencies = frequencies/ np.sum(frequencies)
    
    # Create the bar chart
    plt.figure(figsize=(6, 6))
    plt.bar(unique_values, frequencies, color='red', label='Simulated data')
    
    plt.xlabel(r'Number of dead cows, $N_{{dead \ cows}}$')
    plt.ylabel('Normalized frequency')
    plt.title('Number of Cows Necessary to Make It Across the Field')
    plt.text(0.05, 0.95, f'{number/(size*(size-1)):.2f} Mines per Area', 
             transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='left',
         bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.05, 0.85, f'Field depth={size-2} pixels', 
             transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='left',
         bbox=dict(facecolor='white', alpha=0.5))
    
    # Calculate the mean
    numerator = sum(value*frequency for value, frequency in zip(unique_values, frequencies))
    denominator = sum(frequencies)
    mean = numerator / denominator
    mean = mean
    print(f'The mean is {mean:.2f} iterations.')
    
    # Do a gaussian fit
    def gaussian(x, mu, sigma):
        coefficient = 1 / (np.sqrt(2 * np.pi * sigma ** 2))
        exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        pdf = coefficient * exponent
        return pdf 
   
    initial_mu = np.mean(unique_values)  # mean of your unique_values
    initial_sigma = np.std(unique_values)  # standard deviation of your unique_values

    # Fit the data
    fit = curve_fit(gaussian, unique_values, frequencies, p0=[initial_mu, initial_sigma])
    x_val = np.linspace(0,max(unique_values),1000)
    plt.plot(x_val, gaussian(x_val, fit[0][0], fit[0][1]), 
             label = f'Gaussian with\nμ={fit[0][0]:.0f}, σ={np.abs(fit[0][1]):.0f}')
    plt.legend(loc='upper right')
    
    plt.xlim(mean - 2 * initial_sigma, mean + 2 * initial_sigma)  # Adjust the range as needed

    plt.show()
    
    if save == True :
        plt.savefig(f'Statistics_mine_field(depth={size-2},number={number},sample={sample}).pdf')

def Optimizing_plot(size, sample, save=False):
    """ This function shows the """
    
    number_cows  = [] # Store number of cows
    mine_density = [] # Store mine density per depth (same as per area)
    
    for mine_number in range(size*(size-2)):
        if 100*mine_number % (size*(size-2))==0: # We print each time we do one percent 
            print(f'{round(100*mine_number/(size*(size-2)))}%')
        
        # Count the result
        iteration_list = counter(size=size, number=mine_number, sample=sample)
        iteration_list_count = Counter(iteration_list)
        
        # Extract the result
        unique_values = np.array(sorted(iteration_list_count.keys()))
        frequencies = np.array([iteration_list_count[val] for val in unique_values])
        
        # calculate the mean
        numerator = sum(value*frequency for value,frequency in zip(unique_values,frequencies))
        denominator = sum(frequencies)
        mean = numerator / denominator
        mean = mean
        number_cows.append(mean)
        mine_density.append(mine_number/(size*(size-2)))
    
    # Plot the result
    plt.figure(figsize=(6, 6))
    plt.xlabel(r'Mine density, $\rho_{{mines}}$,  [mines/pixels$^2$]')
    plt.ylabel(r'Number of dead cows, $N_{{dead \ cows}}$')
    plt.title('Number of Dead Cows Versus Mine Density')
    plt.text(0.05, 0.95, f'Field depth = {size-2}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='left',
         bbox=dict(facecolor='white', alpha=0.5))
    
    plt.scatter(mine_density, number_cows, label='Simulated Data')
    
    # Do a linear fit of the data 
    def f(x, A):
        return A*x
    
    fit = curve_fit(f, mine_density, number_cows)
    
    x_val = np.linspace(0,max(mine_density), 1000)
    plt.plot(x_val, f(x_val, fit[0][0]), c='r', 
             label = rf'$N_{{dead \ cows}}$ = {fit[0][0]:.0f} $\rho_{{mines}}$ ')
    
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlim(0,0.1)
    plt.show()
    
    # Optinally save
    if save == True :
        plt.savefig(f'Optimizing_mine_field(depth={size-2},sample={sample}).pdf')
    


#Sactter_animate(size=12, number=60, style='video')

#Statistics_plot(size=22, number=229, sample=10000, save=True)
#Statistics_plot(size=12, number=60,  sample=10000, save=True)

#Optimizing_plot(size=12, sample=100, save=True)
#Optimizing_plot(size=22, sample=100, save=True)
