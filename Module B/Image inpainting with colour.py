import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from collections import Counter

class restoration:
    def __init__(self, picture):
        self.picture = picture
        img = Image.open(picture)
        self.img = img.convert("RGB")
        self.img = np.array(self.img)
                
        c_list = []
        for x in self.img:
            for i in range(len(x)):
                c_list.append(tuple(x[i])) #Create list of all the colours
  
        color_counts = Counter(c_list) #The least common colour is the wrong one
        wrong_color = np.array(color_counts.most_common()[-1][0])
        
        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]))  
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                if np.array_equal(wrong_color, self.img[y, x]): 
                    self.mask[y, x] = 1
                else:
                    self.mask[y, x] = 0

        self.width, self.height = img.size

        if self.width>self.height: #This is just for determinging the size of the pictures
            self.size = 4*2*self.width/self.height, 4*self.height/self.height
        else:
            self.size = 4*2*self.width/self.width, 4*self.height/self.width

    def poisson(self, save=False):
        pix_restored = np.array(self.img)  

        for rgb in range(0, 3):  
            rgb_value = pix_restored[:, :, rgb]
            # Get coordinates of damaged pixels
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
                 A[i, i] = -4
                 for y, x in neighbors:
                    if self.mask[y, x] == 1:  
                        damaged_neighbor_index = np.where((damaged_coord[:,0] == y) &
                                                          (damaged_coord[:,1] == x))[0]
                        A[i,damaged_neighbor_index] = 1 #If neigbhour is damaged, store value
                    else:  
                        b[i]= b[i]-rgb_value[y,x] #If neigbour is'nt damaged, move color value to b matrix 

            restored_pixels = spsolve(A, b) #Solve on form A*c=b, where c are the unkown colours of the damaged pixels

            for i, (y, x) in enumerate(damaged_coord):
                pix_restored[y, x, rgb] = restored_pixels[i] # Replace damaged pixels with restored colour
            
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


# Usage example
picture = "sweden simple.png"  # Replace with your image
restorer = restoration(picture)
restorer.poisson()
