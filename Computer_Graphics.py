from ctypes import c_char
import math
import numpy as np
from numba import jit

def get_greyscale_image(image, colour_wts):
    """
    Gets an image and weights of each colour and returns the image in greyscale
    :param image: The original image
    :param colour_wts: the weights of each colour in rgb (ints > 0)
    :returns: the image in greyscale
    """
    return np.average(image,axis=-1,weights=colour_wts)


    
def reshape_bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros((out_height, out_width, 3))

    scaled_x = float(in_width - 1) / (out_width - 1)
    scaled_y = float(in_height - 1) / (out_height - 1)

    for i in range(out_height):
        for j in range(out_width):
            t = (scaled_x * j) - np.floor(scaled_x * j).astype(int)
            s = (scaled_y * i) - np.floor(scaled_y * i).astype(int)

            c1 = image[np.floor(scaled_y * i).astype(int), np.floor(scaled_x * j).astype(int)]
            c2 = image[np.floor(scaled_y * i).astype(int), np.ceil(scaled_x * j).astype(int)]
            c3 = image[np.ceil(scaled_y * i).astype(int), np.floor(scaled_x * j).astype(int)]
            c4 = image[np.ceil(scaled_y * i).astype(int), np.ceil(scaled_x * j).astype(int)]

            c12 = (1-t) * c1 + t * c2
            c34 = (1-t) * c3 + t * c4
            c = ((1-s) * c12 + s * c34).astype(int)

            new_image[i][j] = c

    new_image = new_image.astype(int)
    return new_image

def gradient_magnitude(image, colour_wts):
    """
    Calculates the gradient image of a given image
    :param image: The original image
    :param colour_wts: the weights of each colour in rgb (> 0) 
    :returns: The gradient image
    """
    m = get_greyscale_image(image, colour_wts)
    gradient_horizontal = m[:, 1:] - m[:, :-1]
    gradient_vertical = m[1:, :] - m[:-1, :]
    gradient_horizontal = gradient_horizontal[:-1, :]
    gradient_vertical = gradient_vertical[:, :-1]
    return (gradient_horizontal**2 + gradient_vertical**2) ** 0.5
    
def visualise_seams(image, new_shape, show_horizontal, colour):
    """
    Visualises the seams that would be removed when reshaping an image to new image (see example in notebook)
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param show_horizontal: the carving scheme to be used.
    :param colour: the colour of the seams (an array of size 3)
    :returns: an image where the removed seams have been coloured.
    """
    greyscale_wt = [0.299, 0.587, 0.114]
    scaled_image = np.copy(image)

    if show_horizontal: # rotate the image
        reversed_scaled_image, list_of_seams = visualise_horizontal_seams(image, new_shape, greyscale_wt)
        reversed_scaled_image = draw_seams(reversed_scaled_image, list_of_seams, colour)
        scaled_image = np.rot90(reversed_scaled_image, 3)

    else: # vertical
        scaled_image, list_of_seams = visualise_vertical_seams(image, new_shape, greyscale_wt)
        scaled_image = draw_seams(scaled_image, list_of_seams, colour)
        
    return scaled_image

def deleteSeam(image, seam_indices):
    'Delete all the pixels through seam from image'
    modified_image = np.copy(image)
    width = image.shape[1]
    for i, j in seam_indices:
        modified_image[i, j:width-1, :] = modified_image[i, j+1:width, :]
    return modified_image[:, :-1, :].astype(np.int32)

def insertSeam(image, seam_indices, colour):
    'Insert all the pixels colour through seam to image'
    height, width, rgb = image.shape
    modified_image = np.zeros((height, width+1, rgb))
    for i, j in seam_indices:
        modified_image[i, 0:j, :] = image[i, 0:j, :]
        modified_image[i, j+1:width+1, :] = image[i, j:width, :] 
        modified_image[i, j] = colour
    return modified_image.astype(np.int32)

def visualise_vertical_seams(image, new_shape, greyscale_wt):
    scaled_image = np.copy(image)
    number_of_seams_to_delete = np.size(image,1) - new_shape[1]
    list_of_seams = []
    for seam_count in range(number_of_seams_to_delete):
        scaled_image, seam_indices = visualise_one_seam(scaled_image, greyscale_wt)
        list_of_seams.append(seam_indices)
    return scaled_image, list_of_seams

def visualise_horizontal_seams(image, new_shape, greyscale_wt):
    reversed_image = np.rot90(image)
    reversed_new_shape = new_shape[::-1]
    reversed_scaled_image, list_of_seams = visualise_vertical_seams(reversed_image, reversed_new_shape, greyscale_wt)
    return reversed_scaled_image, list_of_seams

def draw_seams(scaled_image, list_of_seams, colour):
    'Draw all the pixels colour through seam to image'
    for seam in reversed(list_of_seams):
        scaled_image = insertSeam(scaled_image, seam, colour)
    return scaled_image

def visualise_one_seam(scaled_image, greyscale_wt):
    pixel_energy_matrix = gradient_magnitude(scaled_image, greyscale_wt)
    greyscale_image = get_greyscale_image(scaled_image, greyscale_wt)
    m = np.copy(pixel_energy_matrix)
    height, width = m.shape
    # building cost matrix
    for i in range(1, height):
        for j in range(0, width):
            if j == 0: # handling boundaries
                m[i,j] = pixel_energy_matrix[i,j] + np.amin(np.array(m[i-1,j], m[i-1,j+1] + np.absolute(greyscale_image[i,j+1] - greyscale_image[i-1,j]))) 
            elif j == width - 1: # handling boundaries
                m[i,j] = pixel_energy_matrix[i,j] + np.amin(np.array(m[i-1,j], m[i-1,j-1] + np.absolute(greyscale_image[i-1,j] - greyscale_image[i,j-1])))
            else:
                # costs of new edges that will appear after removal of seam
                c_r = np.absolute(greyscale_image[i,j+1]-greyscale_image[i,j-1]) + np.absolute(greyscale_image[i,j+1] - greyscale_image[i-1,j])
                c_v = np.absolute(greyscale_image[i,j+1]-greyscale_image[i,j-1])
                c_l = np.absolute(greyscale_image[i,j+1]-greyscale_image[i,j-1]) + np.absolute(greyscale_image[i-1,j] - greyscale_image[i,j-1])
                m[i,j] = pixel_energy_matrix[i,j] + np.amin(np.array([m[i-1,j-1] + c_l, m[i-1,j] + c_v, m[i-1,j+1] + c_r]))
    # find the minimum out of the bottom row and get it into a new indices matrix
    seam_indices = np.empty(height, dtype=object)
    min_x = np.argmin(m[-1])
    seam_indices[height-1] = (height-1, min_x)
    # backtracking optimal solution
    for i in reversed(range(height-1)):
        if min_x == 0: # handling boundaries
            if m[i,min_x] == pixel_energy_matrix[i,min_x] + m[i-1,min_x+1] + np.absolute(greyscale_image[i,min_x+1] - greyscale_image[i-1,min_x]):
                min_x = min_x + 1
        elif min_x == width - 1: # handling boundaries
            if m[i,min_x] == pixel_energy_matrix[i,min_x] + m[i-1,min_x-1] + np.absolute(greyscale_image[i-1,min_x] - greyscale_image[i,min_x-1]):
                min_x = min_x - 1
        else:
            # costs of new edges that will appear after removal of seam
            c_r = np.absolute(greyscale_image[i,min_x+1]-greyscale_image[i,min_x-1]) + np.absolute(greyscale_image[i,min_x+1] - greyscale_image[i-1,min_x])
            c_l = np.absolute(greyscale_image[i,min_x+1]-greyscale_image[i,min_x-1]) + np.absolute(greyscale_image[i-1,min_x] - greyscale_image[i,min_x-1])
            if m[i,min_x] == pixel_energy_matrix[i,min_x] + m[i-1,min_x+1] + c_r:
                min_x = min_x + 1
            elif m[i,min_x] == pixel_energy_matrix[i,min_x] + m[i-1,min_x-1] + c_l:
                min_x = min_x - 1    
        seam_indices[i] = (i, min_x)
    scaled_image = deleteSeam(scaled_image, seam_indices)
    return scaled_image, seam_indices

def reshape_seam_crarving(image, new_shape, carving_scheme):
    """
    Resizes an image to new shape using seam carving
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param carving_scheme: the carving scheme to be used.
    :returns: the image resized to new_shape
    """
    greyscale_wt = [0.299, 0.587, 0.114]
    if carving_scheme == 0: # vertical and then horizontal
        # vertical logic
        scaled_image, _ = visualise_vertical_seams(image, new_shape, greyscale_wt)

        # horizontal logic
        reversed_scaled_image, _ = visualise_horizontal_seams(scaled_image, new_shape, greyscale_wt)
        new_image = np.rot90(reversed_scaled_image, 3)

    elif carving_scheme == 1: # horizontal and then vertical
        # horizontal logic
        reversed_scaled_image, _ = visualise_horizontal_seams(image, new_shape, greyscale_wt)
        scaled_image = np.rot90(reversed_scaled_image, 3)
        
        # vertical logic
        new_image, _ = visualise_vertical_seams(scaled_image, new_shape, greyscale_wt)

    else: # intermittent logic
        scaled_image = np.copy(image)
        old_height = np.size(image,0)
        old_width = np.size(image,1)
        number_of_vertical_seams_to_delete = old_width - new_shape[1]
        number_of_horizontal_seams_to_delete = old_height - new_shape[0]

        num_of_smaller_seams_removed = 0
        
        if number_of_vertical_seams_to_delete >= number_of_horizontal_seams_to_delete: # then vertical first
            # vertical logic
            for seam_count in range(number_of_vertical_seams_to_delete):
                scaled_image, _ = visualise_one_seam(scaled_image, greyscale_wt)
                
                # horizontal logic
                if num_of_smaller_seams_removed < number_of_horizontal_seams_to_delete:
                    reversed_image = np.rot90(scaled_image)
                    reversed_new_shape = new_shape[::-1]
                    reversed_scaled_image, _ = visualise_one_seam(reversed_image, greyscale_wt)
                    scaled_image = np.rot90(reversed_scaled_image, 3)
                    num_of_smaller_seams_removed += 1
        else:  
            for seam_count in range(number_of_horizontal_seams_to_delete): # then horizontal first
                # horizontal logic
                reversed_image = np.rot90(scaled_image)
                reversed_new_shape = new_shape[::-1]
                reversed_scaled_image, _ = visualise_one_seam(reversed_image, greyscale_wt)
                scaled_image = np.rot90(reversed_scaled_image, 3)
                
                # vertical logic
                if num_of_smaller_seams_removed < number_of_vertical_seams_to_delete:
                    scaled_image, _ = visualise_one_seam(scaled_image, greyscale_wt)
                    num_of_smaller_seams_removed += 1

    return scaled_image