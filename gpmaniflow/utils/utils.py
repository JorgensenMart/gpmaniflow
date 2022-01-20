import pandas as pd
import numpy as np
from scipy.special import factorial as scipyfac
from scipy.spatial import KDTree
import tensorflow as tf
from gpflow.config import default_float
from gpflow.utilities.ops import square_distance

def factorial(n):
    #print(scipyfac(n))
    return scipyfac(n) # scipy factorial handles arrays
    #return np.prod(range(1,n+1))

def binomial_coef(n, i):
    #print(n)
    #print(i)
    #ni = n - i
    return factorial(n) / (factorial(i) * factorial(n - i))

def bezier_coef(n):
    k = np.ones(n+1)
    nk = np.ones(n+1)
    for i in range(n+1):
        k[i] = factorial(i)
        nk[i] = factorial(n-i)
    return factorial(n) / (k*nk)

def Kronecker(A, B):
        shape = tf.stack([tf.shape(A)[0] * tf.shape(B)[0], tf.shape(A)[1] * tf.shape(B)[1]])
        return tf.reshape(tf.expand_dims(tf.expand_dims(A, 1), 3) * tf.expand_dims(tf.expand_dims(B, 0), 2), shape)

def GetAllListPairs(n, k):
    ''' This function returns all k-lets
    return 
    '''
    I = tf.range(0, n)
    grid = tf.meshgrid(*[I for i in range(k)])
    out = tf.concat([tf.reshape(grid[i], (-1,1)) for i in range(k)], axis = -1)
    return out

def DiscretiseX(X, order):
    ''' 
    This functions returns the grid points near a given sample X 
    X is assumed to be in the hypercube [0,1]^d
    '''
    # VERIFY X IS IN THE HYPERCUBE
    I = tf.math.round(X * order)
    I = tf.py_function(uniq, [I], tf.int32)
    
    return tf.cast(I, dtype = default_float())

import time

def GetNearGridPoints(X, order, n = 100):
    '''
    This function returns the n nearest grid points
    '''
    #VERIFY X IS IN THE HYPERCUBE
    time0 = time.time()
    roundX = tf.math.round(X * order)
    pd = tf.cast(tf.math.floor(n ** (1/tf.shape(X)[1])), default_float())
    time00 = time.time()
    print('Time000:', time00 - time0)
    if pd >= order:
        return tf.cast(GetAllListPairs(order + 1, X.shape[1]), default_float())
    else:
        I = tf.cast(roundX, default_float()) + tf.cast(GetAllListPairs(pd + 1, X.shape[1]), default_float())
        time01 = time.time()
        print('Time0:', time01 - time00)
        if pd + 1 % 2 == 0:
            I = I - pd/2 - tf.cast( tf.math.round(tf.random.uniform(shape = [1])), default_float())
        else:
            I = I - tf.cast(tf.math.floor(pd/2), default_float())
        # MAKE SURE TO TAKE GRID POINTS WITHIN THE CUBE
        maxi = tf.reduce_max(I, axis = 0)
        mini = tf.reduce_min(I, axis = 0)
        #print(maxi)
        #print(tf.math.minimum(mini,0))
        I = I - tf.math.minimum(mini, 0) - tf.math.maximum(maxi - order, 0)
        # RUN A kNN here (probably on numpy) # LIKELY INEFFICIENT, BUT I AM NOT SMARTER
        #tree = KDTree(I.numpy())
        #_, nearIi = tree.query(X, k = n)
        time1 = time.time()
        print('Time1:', time1 - time0)
        minus_distance = - square_distance(X, I) 
        _, nearIi = tf.nn.top_k(input = minus_distance, k = n, sorted = False)
        #print(nearIi)
        time2 = time.time()
        print('Time2:', time2 - time1)
        return tf.gather(I, tf.squeeze(nearIi, axis = 0))



def uniq(J):
    return np.unique(J, axis = 0)

def uniq_keep_order(J):
    J = np.array(J)
    return pd.DataFrame(J).drop_duplicates().values

if __name__ == '__main__':
    #n = 5; k = 6
    #out = GetAllPairs(n, k)
    #print(out)
    X = np.array([[0.1, 0.4, 0.95]])
    I = GetNearGridPoints(X, 10)
    #print(I)
    #print(I.shape)
