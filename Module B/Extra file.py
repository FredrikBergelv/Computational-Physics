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

        pix_d = np.array(self.img)

        # Flatten the pixels into a list of tuples
        pixels = pix_d.reshape(-1, pix_d.shape[-1])
        print('b',pixels)
        pixels_list = [tuple(pixel) for pixel in pixels]
        print('c',pixels)
        
        # Count the occurrences of each color
        color_counts = Counter(pixels_list)

        # Find the least common color (i.e., the most likely "damaged" color)
        wrong_color = np.array(color_counts.most_common()[-1][0])

        # Create a mask based on the least common color (damaged region)
        self.mask = np.zeros((pix_d.shape[0], pix_d.shape[1]))  # Initialize mask
        self.pix_r = np.array(self.img)  # Copy of the original image

        # Identify damaged pixels and mark them in the mask
        for y in range(pix_d.shape[0]):
            for x in range(pix_d.shape[1]):
                if np.array_equal(wrong_color, pix_d[y, x]):
                    self.mask[y, x] = 1
                    self.pix_r[y, x] = [255, 255, 255]  # Mark damaged pixels visually (optional)
                else:
                    self.mask[y, x] = 0

        # Set image dimensions
        self.width, self.height = img.size

        # Determine the size for plotting
        if self.width > self.height:
            self.size = 4 * 2 * self.width / self.height, 4 * self.height / self.height
        else:
            self.size = 4 * 2 * self.width / self.width, 4 * self.height / self.width

    def poisson(self, save=False):
        img_array = np.array(self.img)  # Get the original image array

        # Make a copy to restore damaged regions (for RGB)
        pix_restored = img_array.copy()

        # Separate Poisson restoration for each RGB channel
        for channel in range(3):  # Loop through R (0), G (1), B (2) channels
            print(f"Restoring channel {channel}...")

            # Get the current channel's pixel data
            channel_data = img_array[:, :, channel]

            # Get coordinates of damaged pixels
            damaged_coord = np.argwhere(self.mask == 1)
            n_damaged = len(damaged_coord)

            # Create sparse matrix A and b
            A = csr_matrix((n_damaged, n_damaged))
            b = np.zeros(n_damaged)

            # Loop through all damaged pixels and compute neighbors
            for i, (y, x) in enumerate(damaged_coord):
                neighbors = []
                if y > 0:
                    neighbors.append((y - 1, x))  # Lower neighbor
                if y < self.height - 1:
                    neighbors.append((y + 1, x))  # Upper neighbor
                if x > 0:
                    neighbors.append((y, x - 1))  # Left neighbor
                if x < self.width - 1:
                    neighbors.append((y, x + 1))  # Right neighbor

                A[i, i] = -4
                for ny, nx in neighbors:
                    if self.mask[ny, nx] == 1:  # Damaged neighbor
                        damaged_neighbor_index = np.where((damaged_coord[:, 0] == ny) &
                                                          (damaged_coord[:, 1] == nx))[0]
                        A[i, damaged_neighbor_index] = 1
                    else:  # Non-damaged neighbor
                        b[i] -= channel_data[ny, nx]  # Use current channel's pixel data

            # Solve for the damaged pixels in the current channel
            restored_pixels = spsolve(A, b)

            # Update the current channel with restored pixels
            for i, (y, x) in enumerate(damaged_coord):
                pix_restored[y, x, channel] = restored_pixels[i]

        # Plot the results
        fig, axs = plt.subplots(1, 2, figsize=(self.size))
        axs[0].imshow(self.img)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(pix_restored)  # Show the restored image
        axs[1].set_title("Restored Image (RGB)")
        axs[1].axis('off')

        fig.tight_layout()
        plt.show()

        if save:
            plt.imsave(f'Restored_{self.picture}', pix_restored)

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
