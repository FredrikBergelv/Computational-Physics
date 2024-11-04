import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from collections import Counter

class Jacobian:
    def __init__(self, path):
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
                G = sum(b.Gray for b in self.nb)
                return G / 4  # Return average grayscale value

            def nStep_RGB(self):
                """Compute the average RGB values from neighboring pixels."""
                R = sum(b.R for b in self.nb)
                G = sum(b.G for b in self.nb)
                B = sum(b.B for b in self.nb)
                return (R / 4, G / 4, B / 4)  # Return average RGB values
            
        im = Image.open(path).convert("RGB")  # Load the image and convert it to RGB
        gray = im.convert("L")  # Convert the image to grayscale
        self.size = im.size  # Store image dimensions (width, height)
        color_array = np.array(im)  # Convert image to NumPy array (color)
        gray_array = np.array(gray)  # Convert grayscale image to NumPy array
        self.Gray = gray_array  # Grayscale array
        self.Colour = color_array  # Color array
        self.original = color_array.copy()  # Keep a copy of the original color array

        # Initialize mask to identify bad pixels
        self.mask = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        self.baddies = []  # List of bad pixels

        # Create a matrix of Pixel objects for each pixel in the image
        self.pixels = [[Pixel(color_array[j, i], gray_array[j, i], i, j) for i in range(self.size[0])]
                       for j in range(self.size[1])]
        
    def find_bad(self):
        """Identify bad pixels where the R, G, and B channels differ significantly."""
        gray_tol = 5  # Tolerance for determining grayscale similarity
        index = 1
        for j in range(self.size[1]):  # Loop over image height (rows)
            for i in range(self.size[0]):  # Loop over image width (columns)
                pixel = self.pixels[j][i]
                if not (abs(pixel.R - pixel.G) < gray_tol and abs(pixel.B - pixel.G) < gray_tol and abs(pixel.R - pixel.B) < gray_tol):
                    # If R, G, and B differ beyond the tolerance, mark as bad
                    pixel.bad = 1
                    pixel.bad_index = index  # Assign an index to the bad pixel
                    self.baddies.append(pixel)  # Add bad pixel to the list
                    self.mask[j, i, :] = pixel.RGB  # Mark the pixel in the mask
                    index += 1

    def find_neighbours(self):
        """Assign neighbors to each bad pixel."""
        for pix in self.baddies:
            # Check image boundaries before adding neighbors
            if pix.j + 1 < self.size[1]:  # Bottom neighbor
                pix.nb.append(self.pixels[pix.j + 1][pix.i])
            if pix.j - 1 >= 0:  # Top neighbor
                pix.nb.append(self.pixels[pix.j - 1][pix.i])
            if pix.i + 1 < self.size[0]:  # Right neighbor
                pix.nb.append(self.pixels[pix.j][pix.i + 1])
            if pix.i - 1 >= 0:  # Left neighbor
                pix.nb.append(self.pixels[pix.j][pix.i - 1])

    def set_mask(self, mask_path):
        """Set a mask image and identify bad pixels based on it."""
        mask = Image.open(mask_path).convert("RGB")  # Load the mask image
        mask_size = mask.size
        mask_array = np.array(mask)  # Convert mask image to NumPy array
        im_size = self.size
        x_frac = mask_size[0] / im_size[0]  # Scaling factor for width
        y_frac = mask_size[1] / im_size[1]  # Scaling factor for height

        white_tol = 70  # Tolerance for white pixels
        index = 1
        for j in range(im_size[1]):  # Loop over height
            j = j * y_frac  # Adjust for scaling
            l = round(j)
            for i in range(im_size[0]):  # Loop over width
                i = i * x_frac  # Adjust for scaling
                k = round(i)
                pixel = self.pixels[round(j/y_frac)][round(i/x_frac)]
                if not (abs(mask_array[l, k, 0] - 255) < white_tol and abs(mask_array[l, k, 1] - 255) < white_tol and abs(mask_array[l, k, 2] - 255) < white_tol):
                    # If not a white pixel, mark as bad and update the pixel values
                    pixel.bad = 1
                    pixel.bad_index = index
                    # pixel.R = mask_array[l, k, 0]
                    # pixel.G = mask_array[l, k, 1]
                    # pixel.B = mask_array[l, k, 2]
                    self.baddies.append(pixel)  # Add to bad pixel list
                    self.mask[round(j/y_frac), round(i/x_frac), :] = mask_array[l, k, :]  # Mark in mask
                    index += 1

        # self.update()  # Update the image after applying the mask
        # self.original = self.Colour.copy()  # Save the original color array
        self.find_neighbours()  # Find neighbors of the bad pixels
    
    def destroy_image(self):
        mask = self.mask
        for pix in self.baddies:
            i = pix.i
            j = pix.j
            pix.R = mask[j,i,0]
            pix.G = mask[j,i,1]
            pix.B = mask[j,i,2]
        self.update()  # Update the image after applying the mask
        self.original = self.Colour.copy()  # Save the original color array
        
    def find_mask(self):
        """Find and process the bad pixels based on color differences."""
        self.find_bad()  # Identify bad pixels
        self.find_neighbours()  # Assign neighbors to each bad pixel
        
    def update(self):
        """Update the grayscale and color arrays with the latest pixel values."""
        for j in range(self.size[1]):  # Loop over height
            for i in range(self.size[0]):  # Loop over width
                self.Gray[j, i] = self.pixels[j][i].Gray  # Update grayscale
                self.Colour[j, i, :] = (self.pixels[j][i].R, self.pixels[j][i].G, self.pixels[j][i].B)  # Update color

    def show_all_RBG(self):
        """Display the original and processed color images side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(self.original)  # Show the original image
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(self.Colour)  # Show the processed image
        axes[1].set_title('Processed Image')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    def show_all_Gray(self):
        """Display the original, grayscale, and mask images side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(self.original)  # Show the original image
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(self.Gray, cmap='gray')  # Show the grayscale image
        axes[1].set_title('Processed Image')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    def RGB_process(self, steps, tol, simple=0):
        """Process the RGB values of bad pixels. Only works if the mask was given using the set_mask method."""
        if simple:
            # Quick approach with continueous updates of values
            for n in range(steps):
                max_change = 0  # Track largest change for convergence check
                for s in self.baddies:
                    new_value = s.nStep_RGB()  # Calculate the average RGB from neighbors
                    old_value = (s.R, s.G, s.B)
                    change = max([np.abs(new_value[i] - old_value[i]) / old_value[i] for i in range(3)])  # Calculate change
                    if change > max_change:
                        max_change = change
                    s.R, s.G, s.B = new_value  # Update pixel's RGB values

                if max_change < tol: # Early stopping if the change is small enough
                    print(f'Converged after {n} iterations.')
                    break
        else:
            # Updates all the values in the image at once
            for n in range(steps):
                Tupdate = []
                max_change = 0
                for s in self.baddies:
                    new_value = s.nStep_RGB()
                    old_value = (s.R, s.G, s.B)
                    change = max([np.abs(new_value[i] - old_value[i]) / old_value[i] for i in range(3)])
                    if change > max_change:
                        max_change = change
                    Tupdate.append(new_value)

                for s in self.baddies:
                    s.R, s.G, s.B = Tupdate[s.bad_index - 1]

                if max_change < tol:
                    print(f'Converged after {n} iterations.')
                    break

        self.update()  # Update the image after processing
        print(f"{max_change * 100} % change")
        self.show_all_RBG()  # Show the processed image

    def Gray_process(self, steps, tol, simple=0):
        """Process the grayscale values of bad pixels"""
        if simple:
            for n in range(steps):
                max_change = 0
                for s in self.baddies:
                    new_value = s.nStep_Gray()
                    old_value = s.Gray
                    change = np.abs(new_value - old_value) / old_value  # Calculate change
                    if change > max_change:
                        max_change = change
                    s.Gray = new_value  # Update grayscale value

                if max_change < tol: # Early stopping if the change is small enough
                    print(f'Converged after {n} iterations.')
                    break
        else:
            for n in range(steps):
                Tupdate = []
                max_change = 0
                for s in self.baddies:
                    new_value = s.nStep_Gray()
                    old_value = s.Gray
                    change = np.abs(new_value - old_value) / old_value
                    if change > max_change:
                        max_change = change
                    Tupdate.append(new_value)

                for s in self.baddies:
                    s.Gray = Tupdate[s.bad_index - 1]

                if max_change < tol:
                    print(f'Converged after {n} iterations.')
                    break

        self.update()  # Update the image after processing
        print(f"{max_change * 100} % change")
        self.show_all_Gray()  # Show the processed grayscale image    
        
class Poisson:

    def __init__(self, picture, c=False):
        
        self.c = c
        img = Image.open(picture)
        self.img = img.convert("RGB")
        
        self.array = np.array(self.img) #Get pixels in array with colour
        
        self.mask = np.zeros((self.array.shape[0], self.array.shape[1])) #Form a mask to store if pixle is damaged or not

        self.width, self.height = img.size
        
        #Find the mask when the image is in greyscale
        if c == False:
            for y in range(self.height):    
                for x in range(self.width):               
                    if not self.array[y,x][0] == self.array[y,x][1] == self.array[y,x][2]: #If colour is not grey, then we label as damaged   
                        self.mask[y,x] = 1
                    else:
                        self.mask[y,x] = 0
                        
        #Find the mask when the image is in RGB                
        if c == True:  
            c_list = []
            for x in self.array:
                for i in range(len(x)):
                    c_list.append(tuple(x[i]))
      
            color_counts = Counter(c_list)
            wrong_color = np.array(color_counts.most_common()[-1][0])
            
            self.mask = np.zeros((self.array.shape[0], self.array.shape[1]))  

            for y in range(self.array.shape[0]):
                for x in range(self.array.shape[1]):
                    if np.array_equal(wrong_color, self.array[y, x]):
                        self.mask[y,x] = 1
                    else:
                        self.mask[y,x] = 0
                        
        #This is just for determinging the size of the pictures            
        if self.width>self.height: 
             self.size = 4*2*self.width/self.height, 4*self.height/self.height
        else:
             self.size = 4*2*self.width/self.width, 4*self.height/self.width

    def restore(self, save=False):  


        def chi2(damaged_coord, restored_pixels):
           #This calculates the error χ²
           n = len(damaged_coord) 
           g_pix = self.img.convert("L")
           damaged_pixels = np.array([g_pix.getpixel((x,y)) for y,x in damaged_coord])
           mean = np.mean(restored_pixels)
           sigma2 = (1/(n-1))*np.sum((damaged_pixels-mean)**2) 
           if sigma2 ==0 or n==0 or n==1:
               print("χ² = 0")
           else:
               chi2 = (1 / n)*np.sum(((restored_pixels-damaged_pixels)**2) / sigma2)
               print(f"χ² = {np.round(chi2,3)}") 

        def process(pix_restored, rgb_value=False, rgb=False):
            damaged_coord = np.argwhere(self.mask == 1) #Array of all cooordenates of damaged pixels
        
            A = csr_matrix((len(damaged_coord), len(damaged_coord))) #Form matricies
            b = np.zeros(len(damaged_coord))
        
            for i, (y,x) in enumerate(damaged_coord): #Loop through all damaged pixels
            
                neighbors = [] #We store all exsisting neigbourhs
                if y > 0:  
                    neighbors.append((y-1,x)) #Lower neigbhour
                if y < self.height-1:  
                    neighbors.append((y+1,x)) #Upper neigbhour
                if x > 0:  
                    neighbors.append((y,x-1)) #Left neigbhour
                if x < self.width-1: 
                    neighbors.append((y,x+1)) #Right neigbhour
            
                #We want C_Left + C_Right + C_Upper + C_Lower - 4C_Itself = 0 for all damaged coordinates
                A[i,i] = -4
                for y,x in neighbors:
                    if self.mask[y,x] == 1:  
                        damaged_neighbor_index = np.where((damaged_coord[:,0] == y) & 
                                                      (damaged_coord[:,1] == x))[0]
                        A[i,damaged_neighbor_index] = 1 #If neigbhour is damaged, store value
                    else:
                        if self.c == False:
                            b[i] = b[i]-pix_restored[y,x]  #If neigbour is'nt damaged, move color value to b matrix
                        if self.c == True:
                            b[i] = b[i]-rgb_value[y,x]  #If neigbour is'nt damaged, move color value to b matrix 

            restored_pixels = spsolve(A,b) #Solve on form A*c=b, where c are the unkown colours of the damaged pixels

            for i, (y,x) in enumerate(damaged_coord):
                if self.c == False:
                    pix_restored[y,x] = restored_pixels[i] # Replace damaged pixels with restored colour
                if self.c == True:
                    pix_restored[y,x,rgb] = restored_pixels[i] # Replace damaged pixels with restored colour
                                   
            if self.c == False:
                 chi2(damaged_coord, restored_pixels)

                    
            return pix_restored
        
        
        
        if self.c == False:
            Greyscale = np.array(self.img.convert("L")) #Make a greyscale copy if no colour
            pix_restored = process(Greyscale)
            

                
        if self.c == True:
            RGBscale = np.array(self.array) #Make a copy if colour
            
            for rgb in range(0,3):  

                rgb_value = RGBscale[:,:,rgb]
                
                pix_restored = process(RGBscale,rgb_value,rgb)
        
            
        
        fig, axs = plt.subplots(1,2,figsize=(self.size))    
        axs[0].imshow(self.img)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(pix_restored, cmap='gray')  
        axs[1].set_title("Restored Image")
        axs[1].axis('off')
        
        fig.tight_layout()
        plt.show()
        if save == True : 
            plt.savefig(f'Restored {self.picture}.pdf')
    
  
    def show_damage(self):
        fig, axs = plt.subplots(1,2,figsize=(self.size))  
        axs[0].imshow(self.img)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(self.mask, cmap='binary')
        axs[1].set_title("Damage Localization")
        axs[1].axis('off')
        
        fig.tight_layout()
        plt.show()   

class HeatEquation:
    def __init__(self, picture, k=0.25, h=1, iterations=100):

        self.k = k #Heat eq. constant
        self.h = h #step size
        self.iterations = iterations #number of iterations

        img = Image.open(picture)
        self.img = img.convert("RGB")  
        self.array = np.array(self.img)  
        self.greyscale = np.array(self.img.convert("L"))

        self.mask = np.zeros((self.array.shape[0], self.array.shape[1])) #Create a mask of to store if bad or not

        self.width, self.height = img.size
        
        #If colour is not grey, then we label as damaged    
        for y in range(self.height):
            for x in range(self.width):
                if not self.array[y, x][0] == self.array[y, x][1] == self.array[y, x][2]:  # If color is not grey
                    self.mask[y,x] = 1  
                else:
                    self.mask[y,x] = 0  
        
        #This is just for determinging the size of the pictures            
        if self.width > self.height:
            self.size = 4*2*self.width/self.height, 4*self.height/self.height
        else:
            self.size = 4*2*self.width/self.width, 4*self.height/self.width

    def Heat_eq(self, pix_restored):
        
        def chi2(damaged_coord, pix_restored):
           #This calculates the error χ²
           n = len(damaged_coord) 
           g_pix = self.img.convert("L")
           damaged_pixels = np.array([g_pix.getpixel((x,y)) for y,x in damaged_coord])
           img_restored = Image.fromarray(pix_restored)
           restored_pixels = np.array([img_restored.getpixel((x,y)) for y,x in damaged_coord])
           
           mean = np.mean(restored_pixels)
           sigma2 = (1/(n-1))*np.sum((damaged_pixels-mean)**2) 
           if sigma2 ==0 or n==0 or n==1:
               print("χ² = 0")
           else:
               chi2 = (1 / n)*np.sum(((restored_pixels-damaged_pixels)**2) / sigma2)
               print(f"χ² = {np.round(chi2,3)}") 
               
               
        damaged_coords = np.argwhere(self.mask == 1) #Find the coord. where we have damaged pixels 
        temp_colour = pix_restored  

        #Iteration for the heat equation
        for iteration in range(self.iterations):
            for (y, x) in damaged_coords:
                neighbours = []
                k = self.k
                h = self.h
                
                if y > 0:  
                    neighbours.append(pix_restored[y-1, x]) #Lower neighbour
                if y < self.height - 1:  
                    neighbours.append(pix_restored[y+1, x]) #Upper neighbour
                if x > 0:  
                    neighbours.append(pix_restored[y, x-1]) #Left neighbour
                if x < self.width - 1:  
                    neighbours.append(pix_restored[y, x+1]) #Right neighbour

                #Here we state that each damaged pixel should be the average of all it's neigbhours
                temp_colour[y,x] = pix_restored[y,x] + k*h*(np.sum(neighbours) - 4*pix_restored[y,x])

            pix_restored = temp_colour #Save the last iteration
            
        chi2(damaged_coords, pix_restored)

        return pix_restored

    def restore(self, save=False):

               
        
        #Use the heat equation
        pix_restored = self.Heat_eq(self.greyscale)

        fig, axs = plt.subplots(1, 2, figsize=(self.size))
        axs[0].imshow(self.img)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(pix_restored, cmap='gray')
        axs[1].set_title("Restored Image")
        axs[1].axis('off')

        fig.tight_layout()
        plt.show()

        if save:
            Image.fromarray(pix_restored.astype(np.uint8)).save(f"Restored_{self.picture}.png")



# Function to call for the two different methods 
def restore(path, method, ifcolour=False, mask=False):
    if method == Jacobian:
        if ifcolour == False:
            picture = Jacobian(path)
            picture.find_mask()
            picture.Gray_process(steps=100, tol=0.0001, simple=1) # Perform grayscale processing
        if ifcolour == True: #Uses a layer as the mask
             pic = Jacobian(path)
             pic.set_mask(mask)  # Set mask to identify bad pixels. The mask must have a white bachground
             pic.destroy_image() # Put the mask image onto the original image
             pic.RGB_process(steps=400, tol=0.0001, simple=1)  # Perform RGB processing
      
    if method == Poisson:
        picture = Poisson(path, ifcolour) #Uses the least common colour as mask
        picture.restore()
        picture.show_damage()

    if method == HeatEquation:
        picture = HeatEquation(path, k=1/2, h=1/2) #Uses the least common colour as mask
        picture.restore()
    
    
restore("Lund.png", Poisson, ifcolour=True)
