# Function that linearly interpolates two points
import numpy as np
#import tensorflow as tf

def linear_interpolate(z1, z2, mesh = 10):
    D = z1.shape[1]
    diff = z2 - z1 #
    zerotoone = np.tile(np.linspace(0,1, mesh), (D,1)).T
    out = z1 + diff * zerotoone
    return out

#z1 = np.array([[0 , 1]]); z2 = np.array([[0, 2]])

#linear_interpolate(z1,z2)

#def quadrature():
    