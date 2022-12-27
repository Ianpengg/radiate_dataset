import cv2
import numpy as np 
from copy import copy

def plot_grid(img: np.ndarray , grid_size: int) -> np.ndarray:
    '''
    Add grid on the image 

   
    @param img: Image
    @param grid_size: int  
    '''
    new_img = copy(img)
    height, width, channels = new_img.shape
    for x in range(0, width -1, grid_size):
        cv2.line(new_img, (x, 0), (x, height), (100, 100, 100), 1, 1)
        cv2.line(new_img, (0,x), (height, x), (100, 100, 100), 1, 1)
    cv2.imshow("with_grid", new_img)