import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix


class restoration:
    def __init__(self, picture):
        
        self.picture = picture
        img = Image.open(picture)
        self.img = img.convert("RGB")

        pix_original = np.array(self.img) #Get pixels in array with colour

        #Form a mask to store if pixle is damaged or not
        self.mask = np.zeros((pix_original.shape[0], pix_original.shape[1])) 

        self.width, self.height = img.size

         #If colour is not grey, then we label as damaged
        for y in range(self.height): 
            for x in range(self.width):                 
                if not pix_original[y, x][0] == pix_original[y, x][1] == pix_original[y, x][2]:
                    self.mask[y, x] = 1
                else:
                    self.mask[y, x] = 0
                    
        if self.width>self.height: #This is just for determinging the size of the pictures
             self.size = 4*2*self.width/self.height, 4*self.height/self.height
        else:
             self.size = 4*2*self.width/self.width, 4*self.height/self.width

    def poisson(self, save=False):
        pix_restored = np.array(self.img.convert("L")) #Make a greyscale copy
        
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
            for y, x in neighbors:
                if self.mask[y,x] == 1:  
                    damaged_neighbor_index = np.where((damaged_coord[:,0] == y) & 
                                                      (damaged_coord[:,1] == x))[0]
                    A[i,damaged_neighbor_index] = 1 #If neigbhour is damaged, store value
                else:
                    b[i] = b[i]-pix_restored[y,x]  #If neigbour is'nt damaged, move color value to b matrix

        restored_pixels = spsolve(A,b) #Solve on form A*c=b, where c are the unkown colours of the damaged pixels

        for i, (y, x) in enumerate(damaged_coord):
            pix_restored[y, x] = restored_pixels[i] # Replace damaged pixels with restored colour
            
        fig, axs = plt.subplots(1, 2, figsize=(self.size))    
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
    
        #This calculates the error χ²
        n = len(damaged_coord) 
        g_pix = self.img.convert("L")
        damaged_pixels = np.array([g_pix.getpixel((x, y)) for y, x in damaged_coord])
        mean = np.mean(damaged_pixels)
        sigma2 = (1/(n-1))*np.sum((damaged_pixels-mean)**2) 
        if sigma2 ==0 or n==0 or n==1:
            print("χ² = ∞")
        else:
            chi2 = (1 / n)*np.sum(((restored_pixels-damaged_pixels)**2) / sigma2)
            print(f"χ² = {np.round(chi2,3)}")      
            
        
    def show_damage(self):
        fig, axs = plt.subplots(1, 2, figsize=(self.size))  
        axs[0].imshow(self.img)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(self.mask, cmap='binary')
        axs[1].set_title("Damage Localization")
        axs[1].axis('off')
        
        fig.tight_layout()
        plt.show()   

#Picture = restoration("Wilhelm.png")
#Picture = restoration("Prussia Damaged.png")
#Picture = restoration("Kungen.png")
#Picture = restoration("Film.png")
Picture = restoration("Kungen2.png") #0,1,2,3,4,5
#plain = 'plain.png', 'plain1.png', 'plain2.png', 'plain3.png'


Picture.poisson()

