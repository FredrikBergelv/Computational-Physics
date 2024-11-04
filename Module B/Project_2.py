import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from collections import Counter

class Pixel:
    def __init__(self, colour, gray, i, j):
        # Store RGB values and grayscale value
        self.R = colour[0]
        self.G = colour[1]
        self.B = colour[2]
        self.RGB = (colour[0], colour[1], colour[2])  # Tuple for RGB values
        self.Gray = gray  # Grayscale value
        self.j = j  # y-coordinate (row)
        self.i = i  # x-coordinate (column)
        self.bad_index = 0  # Index for identifying bad pixels
        self.bad = 0  # Flag to mark if the pixel is 'bad' (non-grayscale)
        self.nb = []  # List to store neighboring pixels
        
    def nStep_Gray(self):
        """Compute the average grayscale value from neighboring pixels."""
        w = 1
        h = 0
        G = w/4*(sum(b.Gray for b in self.nb)-h**2)# + (1-w)*self.Gray
        return G  # Return average grayscale value

    def nStep_RGB(self):
        """Compute the average RGB values from neighboring pixels."""
        w = 1
        h = 0
        R = w/4*(sum(b.R for b in self.nb)-h**2) + (1-w)*self.R 
        G = w/4*(sum(b.G for b in self.nb)-h**2) + (1-w)*self.G
        B = w/4*(sum(b.B for b in self.nb)-h**2) + (1-w)*self.B
        return (R , G , B )  # Return average RGB values
    
    def Step_heat(self,k,h):
        G = self.Gray + k*h*(sum(b.Gray for b in self.nb) - 4*self.Gray)
        return G
        
class Joined:
    def __init__(self, path,c = 0,save = False, number = 2):
            
        im = Image.open(path).convert("RGB")  # Load the image and convert it to RGB
        gray = im.convert("L")  # Convert the image to grayscale
        self.size = im.size  # Store image dimensions (width, height)
        self.width, self.height = im.size
        color_array = np.array(im)  # Convert image to NumPy array (color)
        gray_array = np.array(gray)  # Convert grayscale image to NumPy array
        self.Gray = gray_array  # Grayscale array
        self.Colour = color_array  # Color array
        self.original = color_array.copy()  # Keep a copy of the original color array
        self.original_gray = gray_array.copy()

        # Initialize mask to identify bad pixels
        self.mask = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        self.mask2 = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        self.baddies = []  # List of bad pixels

        # Create a matrix of Pixel objects for each pixel in the image
        self.pixels = [[Pixel(color_array[j, i], gray_array[j, i], i, j) for i in range(self.size[0])]
                       for j in range(self.size[1])]
        self.c = c
        self.A = np.array([]) # Store the matrices
        self.b = np.array([]) #
        self.save = save
        
        self.steps = 400 # Number of iterations
        self.tol = 0.5 # Tolerance for early stopping
        self.simple = 1 # Wether to do instant updates
        
        self.k = 0.25
        self.h = 1
        
        self.path = path
        
        colour_tolerance = 10
        # Find the mask when the image is in grayscale
        if c == False:
            count = 1
            for y in range(self.height):    
                for x in range(self.width): 
                    pixel = self.pixels[y][x]
                    # If color is not grey, then we label it as damaged   
                    if not self.Colour[y,x][0] == self.Colour[y,x][1] == self.Colour[y,x][2]: 
                        self.mask[y,x] = 1
                        self.mask2[y,x,:] = pixel.RGB
                        self.baddies.append(pixel)
                        pixel.bad = 1
                        pixel.bad_index = count
                        count += 1
                        
        # Find the mask when the image is in RGB                
        if c == True:  
            count = 1
            c_list = []
            for x in self.original:
                for i in range(len(x)):
                    c_list.append(tuple((round((x[i][0]/colour_tolerance))*colour_tolerance,
                                         round((x[i][1]/colour_tolerance))*colour_tolerance,
                                         round((x[i][2]/colour_tolerance))*colour_tolerance)))
      
            color_counts = Counter(c_list)
            rcolor = (color_counts.most_common(number))
            right_color = [a[0] for a in rcolor]

            for y in range(self.original.shape[0]):
                for x in range(self.original.shape[1]):
                    good = 0
                    pixel = self.pixels[y][x]
                    for col in right_color:
                        # Check if pixel color matches one of the common colors
                        if not (abs(col[0] - self.original[y, x][0]) < colour_tolerance and 
                            abs(col[1] - self.original[y, x][1]) < colour_tolerance and 
                            abs(col[2] - self.original[y, x][2]) < colour_tolerance):
                            good += 1
                    # If the pixel is damaged (doesn't match any common colors)
                    if good == number:   
                        self.mask[y,x] = 1
                        self.mask2[y,x,:] = pixel.RGB
                        self.baddies.append(pixel)
                        pixel.bad = 1
                        pixel.bad_index = count
                        count += 1
                    else:
                        self.mask[y,x] = 0
    
        self.find_neighbours()
        
    def update(self):
        """Update the grayscale and color arrays with the latest pixel values."""
        for j in range(self.size[1]):  # Loop over height
            for i in range(self.size[0]):  # Loop over width
                self.Gray[j, i] = self.pixels[j][i].Gray  # Update grayscale
                self.Colour[j, i, :] = (self.pixels[j][i].R, self.pixels[j][i].G, self.pixels[j][i].B) 

    def show_all_RBG(self):
        """Display the original and processed color images side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create subplots for comparison
        axes[0].imshow(self.original)  # Show the original image on the left
        axes[0].set_title('Original Image')  # Title for the original image
        axes[0].axis('off')  # Hide axis lines and labels
        axes[1].imshow(self.Colour)  # Show the processed image on the right
        axes[1].set_title('Processed Image')  # Title for the processed image
        axes[1].axis('off')  # Hide axis lines and labels
        plt.tight_layout()  # Adjust layout for better spacing
        if self.save == True:  # If save flag is True, save the image
            plt.savefig(f'Restored.png')  # Save the image as 'Restored.png'
        else:
            plt.show()  # If not saving, display the image

    def show_all_Gray(self):
        """Display the original, grayscale, and mask images side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create subplots for comparison
        axes[0].imshow(self.original)  # Show the original image on the left
        axes[0].set_title('Original Image')  # Title for the original image
        axes[0].axis('off')  # Hide axis lines and labels
        axes[1].imshow(self.Gray, cmap='gray')  # Show the grayscale image on the right
        axes[1].set_title('Processed Image')  # Title for the processed image
        axes[1].axis('off')  # Hide axis lines and labels
        # The following lines are commented out but were intended to show a mask
        # axes[2].imshow(self.mask2)  # Show the mask image (commented out)
        # axes[2].set_title('Mask')
        # axes[2].axis('off')
        plt.tight_layout()  # Adjust layout for better spacing
        if self.save == True:  # If save flag is True, save the image
            plt.savefig(f'Restored.png')  # Save the image as 'Restored.png'
        else:
            plt.show()  # If not saving, display the image

    def chi2(self):
        """This calculates the error χ² (chi-squared)."""
        n = len(self.baddies)  # Get the number of damaged pixels (baddies)
        g_pix = self.Gray  # Grayscale pixel data
        damaged_coords = np.array([(pix.i, pix.j) for pix in self.baddies])  # Get coordinates of damaged pixels
        restored_pixels = np.array([g_pix[coord[1], coord[0]] for coord in damaged_coords])  # Get restored pixel values
        damaged_pixels = np.array([self.original_gray[coord[1], coord[0]] for coord in damaged_coords])  # Original pixel values before damage
        mean = np.mean(restored_pixels)  # Calculate the mean of restored pixels
        sigma2 = (1/(n-1)) * np.sum((damaged_pixels-mean)**2)  # Calculate variance
        if sigma2 == 0 or n == 0 or n == 1:  # Check for trivial cases where chi-squared is 0
            print("χ² = 0")
        else:
            chi2 = (1 / n) * np.sum(((restored_pixels-damaged_pixels)**2) / sigma2)  # Compute chi-squared value
            print(f"χ² = {np.round(chi2, 3)}")  # Print rounded chi-squared value

    def find_neighbours(self):
        """Find the neighboring pixels for each damaged pixel."""
        for pix in self.baddies:  # Loop through all damaged pixels
            x = pix.i  # x-coordinate of the pixel
            y = pix.j  # y-coordinate of the pixel
            i = pix.bad_index - 1  # Bad pixel index
            
            neighbors = []  # List to store existing neighbors
            if y > 0:  # Check lower neighbor
                pix.nb.append(self.pixels[y-1][x])  # Append lower neighbor
            if y < self.height - 1:  # Check upper neighbor
                pix.nb.append(self.pixels[y+1][x])  # Append upper neighbor
            if x > 0:  # Check left neighbor
                pix.nb.append(self.pixels[y][x-1])  # Append left neighbor
            if x < self.width - 1:  # Check right neighbor
                pix.nb.append(self.pixels[y][x+1])  # Append right neighbor

    def restore(self): 
        """Restoration of damaged pixels using the Poisson method."""
        
        def process(pix_restored, rgb_value=False, rgb=0):
            """Sub-function to handle processing of each pixel."""
            damaged_coord = np.argwhere(self.mask == 1)  # Get coordinates of damaged pixels
            A = csr_matrix((len(self.baddies), len(self.baddies)))  # Initialize sparse matrix A
            b = np.zeros(len(self.baddies))  # Initialize vector b
            
            for pix in self.baddies:  # Loop through damaged pixels
                x = pix.i
                y = pix.j
                i = pix.bad_index - 1
                A[i, i] = -4  # Set diagonal element of matrix A
                for pixe in pix.nb:  # Loop through pixel neighbors
                    x = pixe.i
                    y = pixe.j
                    if pixe.bad:  # If neighbor is damaged, store value in A
                        A[i, pixe.bad_index - 1] = 1
                    else:
                        if self.c == False:  # If not color, update b for grayscale
                            b[i] = b[i] - pix_restored[y, x]  
                        if self.c == True:  # If color, update b for color image
                            b[i] = b[i] - pix_restored[y, x]

            self.A = A  # Store matrix A
            self.b = b  # Store vector b
                
            restored_pixels = spsolve(A, b)  # Solve the system of linear equations

            for pix in self.baddies:  # Loop through damaged pixels to restore
                if self.c == False:  # Grayscale case
                    pix_restored[pix.j, pix.i] = restored_pixels[pix.bad_index - 1]  
                if self.c == True:  # Color case
                    pix_restored[pix.j, pix.i] = restored_pixels[pix.bad_index - 1]
                                   
            if self.c == False:  # If grayscale, compute chi-squared error
                   self.chi2()
                    
            return pix_restored  # Return the restored pixel data
        
        if self.c == False:  # Grayscale restoration
            Greyscale = self.Gray.copy()  # Make a copy of the grayscale image
            Greyscale = process(Greyscale)  # Process the grayscale image
            self.Gray = Greyscale  # Update the image
            self.show_all_Gray()  # Display the original and processed grayscale images

        if self.c == True:  # Color restoration
            RGBscale = np.array(self.Colour.copy())  # Make a copy of the color image
                        
            for rgb in range(0, 3):  # Process each color channel (R, G, B)
                rgb_value = RGBscale[:, :, rgb]  # Get one color channel
                self.Colour[:, :, rgb] = process(rgb_value)  # Restore the color channel
            self.show_all_RBG()  # Show the original and processed color images

    def RGB_process(self, simple):  # "Jacobian method for colour"
        """Process the RGB values of bad pixels. Only works if the mask was given using the set_mask method."""
        steps = self.steps  # Number of iterations (steps) allowed for convergence
        tol = self.tol  # Tolerance level for convergence
        if simple:
            # Quick approach with continuous updates of values (simple Jacobi method)
            for n in range(steps):
                max_change = 0  # Track the largest change for convergence check
                for s in self.baddies:
                    new_value = s.nStep_RGB()  # Calculate the average RGB from neighbors
                    old_value = (s.R, s.G, s.B)  # Store old RGB values
                    change = max([np.abs(new_value[i] - old_value[i]) for i in range(3)])  # Calculate change for R, G, and B channels
                    if change > max_change:
                        max_change = change  # Update maximum change if necessary
                    s.R, s.G, s.B = new_value  # Update pixel's RGB values with the new value

                if max_change < tol:  # Stop early if the change is small enough (converged)
                    print(f'Converged after {n} iterations.')
                    break
        else:
            # Update all the values in the image at once (more complex approach)
            for n in range(steps):
                Tupdate = []  # Temporary list to store updates for each pixel
                max_change = 0  # Track the largest change
                for s in self.baddies:
                    new_value = s.nStep_RGB()  # Calculate the average RGB from neighbors
                    old_value = (s.R, s.G, s.B)  # Store old RGB values
                    change = max([np.abs(new_value[i] - old_value[i]) for i in range(3)])  # Calculate change for R, G, and B channels
                    if change > max_change:
                        max_change = change  # Update maximum change if necessary
                    Tupdate.append(new_value)  # Append new RGB value to the list

                for s in self.baddies:
                    s.R, s.G, s.B = Tupdate[s.bad_index - 1]  # Update each pixel's RGB values from Tupdate

                if max_change < tol:  # Stop early if the change is small enough (converged)
                    print(f'Converged after {n} iterations.')
                    break

        self.update()  # Update the image after processing
        print(f"{max_change} change")  # Print the final change
        self.show_all_RBG()  # Show the original and processed color images

    def Gray_process(self, simple):  # "Jacobian method for grayscale"
        """Process the grayscale values of bad pixels."""
        steps = self.steps  # Number of iterations (steps) allowed for convergence
        tol = self.tol  # Tolerance level for convergence
        if simple:
            # Simple Jacobi method with continuous updates
            for n in range(steps):
                max_change = 0  # Track the largest change
                for s in self.baddies:
                    new_value = s.nStep_Gray()  # Calculate the new grayscale value from neighbors
                    old_value = s.Gray  # Store the old grayscale value
                    change = np.abs(new_value - old_value)  # Calculate change in grayscale value
                    if change > max_change:
                        max_change = change  # Update maximum change if necessary
                    s.Gray = new_value  # Update pixel's grayscale value

                if max_change < tol:  # Stop early if the change is small enough (converged)
                    print(f'Converged after {n} iterations.')
                    break
        else:
            # Update all values at once
            for n in range(steps):
                Tupdate = []  # Temporary list to store updates for each pixel
                max_change = 0  # Track the largest change
                for s in self.baddies:
                    new_value = s.nStep_Gray()  # Calculate the new grayscale value from neighbors
                    old_value = s.Gray  # Store the old grayscale value
                    change = np.abs(new_value - old_value)  # Calculate change in grayscale value
                    if change > max_change:
                        max_change = change  # Update maximum change if necessary
                    Tupdate.append(new_value)  # Append new grayscale value to the list

                for s in self.baddies:
                    s.Gray = Tupdate[s.bad_index - 1]  # Update each pixel's grayscale value from Tupdate

                if max_change < tol:  # Stop early if the change is small enough (converged)
                    print(f'Converged after {n} iterations.')
                    break

        self.update()  # Update the image after processing
        print(f"{max_change} change")  # Print the final change
        self.show_all_Gray()  # Show the original and processed grayscale images
        self.chi2()  # Calculate chi-squared error

    def Heat_eq(self, k=1/2, h=1/2):
        """Iterative method for solving the heat equation to restore damaged pixels."""
        if self.simple:  # Simple iterative method
            for n in range(self.steps):
                Tupdate = []  # Temporary list to store updates for each pixel
                max_change = 0  # Track the largest change
                for s in self.baddies:
                    new_value = s.Step_heat(k, h)  # Calculate new value based on heat equation
                    old_value = s.Gray  # Store old grayscale value
                    change = np.abs(new_value - old_value)  # Calculate change in grayscale value
                    if change > max_change:
                        max_change = change  # Update maximum change if necessary
                    s.Gray = new_value  # Update pixel's grayscale value

                if max_change < self.tol:  # Stop early if the change is small enough (converged)
                    print(f'Converged after {n} iterations.')
                    break
        else:  # More complex update-all-at-once approach
            for n in range(self.steps):
                Tupdate = []  # Temporary list to store updates for each pixel
                max_change = 0  # Track the largest change
                for s in self.baddies:
                    new_value = s.Step_heat(k, h)  # Calculate new value based on heat equation
                    old_value = s.Gray  # Store old grayscale value
                    change = np.abs(new_value - old_value)  # Calculate change in grayscale value
                    if change > max_change:
                        max_change = change  # Update maximum change if necessary
                    Tupdate.append(new_value)  # Append new grayscale value to the list

                for s in self.baddies:
                    s.Gray = Tupdate[s.bad_index - 1]  # Update each pixel's grayscale value from Tupdate

                if max_change < self.tol:  # Stop early if the change is small enough (converged)
                    print(f'Converged after {n} iterations.')
                    break

        self.update()  # Update the image after processing
        print(f"{max_change} change")  # Print the final change
        self.chi2()  # Calculate chi-squared error
        self.show_all_Gray()  # Show the original and processed grayscale images

    def show_damage(self):
        """Display the original image and damage localization side by side."""
        fig, axs = plt.subplots(1, 2, figsize=(self.size))  # Create subplots for comparison
        axs[0].imshow(self.original)  # Show the original image on the left
        axs[0].set_title("Original Image")  # Title for the original image
        axs[0].axis('off')  # Hide axis lines and labels

        axs[1].imshow(self.mask2)  # Show the damage localization on the right
        axs[1].set_title("Damage Localization")  # Title for the damage localization
        axs[1].axis('off')  # Hide axis lines and labels
        
        fig.tight_layout()  # Adjust layout for better spacing
        plt.show()  # Display the subplots

    def fix(self, method):
        """Fix the damaged image based on the specified method.
        method:
        0 = Poisson (matrix inversion)
        1 = Poisson (Jacobi method)
        2 = Heat equation
        """
        if method == 0:
            self.restore()  # Call the Poisson method (matrix inversion)
        elif method == 1:
            if self.c == 0:
                self.Gray_process(self.simple)  # Call grayscale Jacobi method if image is grayscale
            else:
                self.RGB_process(self.simple)  # Call RGB Jacobi method if image is color
        elif method == 2:
            self.Heat_eq(self.k, self.h)  # Call heat equation restoration method

#%% 
import time  # For measuring execution time

# Initialize the picture
picture = Joined("flag5.png", c=1, number=3)  # Initialize with image file and mask settings
# picture.show_damage()  # Display the original image and damage localization

##%% Measure the time for processing
start_time = time.time()  # Start the timer
# Set parameters
picture.steps = 40  # Maximum number of iterations
picture.tol = 0.01  # Tolerance for maximum change in values for pixels for early stopping
picture.save = 0  # Flag for saving the result as an image
picture.simple = 1  # Whether to use the simple method
picture.k = 1/4  # Set k for heat equation
picture.fix(1)  # Use method x to fix the image
# 0 = Poisson (matrix inversion)
# 1 = Poisson (Jacobi method)
# 2 = Heat equation
print("--- %s seconds ---" % (time.time() - start_time))  # Print the elapsed time
