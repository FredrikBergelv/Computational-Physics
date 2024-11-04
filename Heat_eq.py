import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

class HeatEquation:
    def __init__(self, picture, k=1/2, h=1/2, iterations=100):

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
        damaged_coords = np.argwhere(self.mask == 1) #Find the coord. where we have damaged pixels 
        temp_colour = np.copy(pix_restored)
        counter = 0


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
            
            
            if np.all(pix_restored- temp_colour == 0):
                    counter += 1  
            else:
                        counter = 0  
                    
            if counter >= 3:
                    print(f"Converged after {iteration} iterations.")
                    return pix_restored
            pix_restored = np.copy(temp_colour) #Save the last iteration

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


#Note that k*n<0.5
heat_eq = HeatEquation("OscarII.png", k=1/2, h=1/2, iterations=100)
heat_eq.restore()  
