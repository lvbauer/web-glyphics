import numpy as np
import math


def calc_green_index(r, g, b):
    
    preGI_b = ((255 - abs(g-165)) + (255-abs(r-37.5)) + (255-abs(b-37.5))) / (3*255)
    preGI_c = preGI_b / (1-preGI_b)
    green_index = preGI_c / 12

    return green_index


def calc_green_index_array(rgb_arr):

    r = rgb_arr[:,:,0]
    g = rgb_arr[:,:,1]
    b = rgb_arr[:,:,2]

    return calc_green_index(r, g, b)

