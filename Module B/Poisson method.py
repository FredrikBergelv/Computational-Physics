"""
Created on Sat Sep 28 21:39:57 2024

@author: Fredrik Bergelv
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from collections import Counter


class Poisson_restoration:
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
           mean = np.mean(damaged_pixels)
           sigma2 = (1/(n-1))*np.sum((damaged_pixels-mean)**2) 
           if sigma2 ==0 or n==0 or n==1:
               print("χ² = ∞")
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


 
#Picture = Poisson_restoration("Wilhelm.png")
#Picture = Poisson_restoration("Prussia Damaged.png")
#Picture = Poisson_restoration("Kungen.png")
#Picture = Poisson_restoration("Film.png")
#Picture = Poisson_restoration("Kungen2.png") #0,1,2,3,4,5
#plain = 'plainPoisson_restorationpng', 'plain1.png', 'plain2.png', 'plain3.png'


Picture = Poisson_restoration('sweden simple.png', c=True )


Picture.restore()

