#import pandas as pd
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
    
def log_beta_function(z1, z2):
    return tf.math.lgamma(z1) + tf.math.lgamma(z2) - tf.math.lgamma(z1 + z2)

def xlog(x):
    return tf.where(tf.equal(x, tf.zeros_like(x)), tf.cast(tf.ones_like(x), default_float()), x * tf.cast(tf.math.log(x), default_float()))

def BetaGammaIntegral(gamma, order, i): 
    logconst = - (-xlog(order) + xlog(i) + xlog(order - i) + tf.cast(tf.math.log(binomial_coef(order,i)), default_float())) * (gamma-1)
    logconst2 = tf.cast(tf.math.log(binomial_coef(order, i)), default_float()) * gamma
    logintegral = log_beta_function(gamma*i + 1, gamma*(order - i) + 1)
    out = tf.math.exp(logintegral + logconst + logconst2)
    return out

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
    #time0 = time.time()
    roundX = tf.math.round(X * order)
    pd = tf.cast(tf.math.floor(n ** (1/tf.shape(X)[1])), default_float())
    #time00 = time.time()
    #print('Time000:', time00 - time0)
    if pd >= order:
        return tf.cast(GetAllListPairs(order + 1, X.shape[1]), default_float())
    else:
        I = tf.cast(roundX, default_float()) + tf.cast(GetAllListPairs(pd + 1, X.shape[1]), default_float())
        #time01 = time.time()
        #print('Time0:', time01 - time00)
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
        #time1 = time.time()
        #print('Time1:', time1 - time0)
        minus_distance = - square_distance(X, I) 
        _, nearIi = tf.nn.top_k(input = minus_distance, k = n, sorted = False)
        #print(nearIi)
        #time2 = time.time()
        #print('Time2:', time2 - time1)
        return tf.gather(I, tf.squeeze(nearIi, axis = 0))

def GetNearGridPoints2(X, order, d, n = 100):
    roundX = tf.math.round(X * order)
    #roundX = tf.math.floor(X * order)
    print(roundX)
    zeroX = X * order - roundX # Should be "close" to origo
    print(zeroX)
    if (order + 1) ** d < n:
        n = (order + 1) ** d
        pd = order
    else:
        print("hey")
        pd = tf.cast(tf.math.floor(n ** (1/d)), default_float())
    print(pd) 
    #if pd >= order:
    #    return tf.cast(tf.tile(GetAllListPairs(order + 1, d)[None,:,:], [order + 1, 1, 1]), default_float())
    #else:
    I = tf.cast(GetAllListPairs(pd + 1, d), default_float())
    print((pd + 1) % 2)
    if (pd + 1) % 2 == 0: # pd is uneven
        print("hey")
        I = I - (pd-1)/2 - tf.cast( tf.math.round(tf.random.uniform(shape = [d])), default_float())
    else: # pd is even
        I = I - tf.cast(tf.math.floor(pd/2), default_float())
    print(I)
    print(I + tf.expand_dims(roundX, axis = 1))
    mini = tf.reduce_min(I + tf.expand_dims(roundX, axis = 1), axis = 1)
    maxi = tf.reduce_max(I + tf.expand_dims(roundX, axis = 1), axis = 1)
    zeroX = zeroX + tf.math.minimum(mini,0) + tf.math.maximum(maxi - order, 0)
    print(zeroX)
    minus_distance = - square_distance(zeroX, I)
    print(minus_distance)
    #print(I)
    _, nearIi = tf.nn.top_k(input = minus_distance, k = n, sorted = True)

    out = tf.gather(I, nearIi)
    #out = out + tf.expand_dims(roundX, axis = 1)
    out = out + tf.expand_dims(roundX - tf.math.minimum(mini,0) - tf.math.maximum(maxi - order, 0), axis = 1)

    #maxi = tf.reduce_max(out, axis = 1)
    #mini = tf.reduce_min(out, axis = 1)
    #print(maxi)
    #print(mini)
    #out = out - tf.expand_dims(
    #        tf.math.minimum(mini, 0) + tf.math.maximum(maxi - order, 0),
    #        axis = 1) # Ensures points within hypercube
    
    #print(out) 
    return out

def GetNearGridPoints3(BXnew, input_dim, order, n = 100):
    # BXnew is [N, d, order + 1]
    def GetBiggestSubSum(A, n = n):
        # A is [d, order + 1]
        #print(A)
        idx = tf.argsort(A, axis = 1, direction = 'DESCENDING')
        #print(idx)
        #As = tf.gather(A, idx, axis = 1)
        #print(As)
        #I = tf.ragged.constant( [[0]] * input_dim)
        #I = tf.gather(A, tf.gather(idx, [0] * input_dim, batch_dims = 1), batch_dims = 1)
        #print(I)
        #I = tf.ragged.constant(I)
        #print(I)
        NumberOfPoints = 1; Att = tf.constant([1] * input_dim, dtype = tf.int32)
        #print(Att)
        #while tf.less(NumberOfPoints, n):
        def foo(Att): 
            M = tf.gather(A, tf.gather(idx, Att, batch_dims = 1), batch_dims = 1)
            #print(new)
            #new = tf.argsort(M)[-1]
            def recursive_check(M, Att, i):
                #print(i)
                new = tf.argsort(M)[i]
                #print(Att)
                if tf.math.equal(Att[new], order):
                    if i == -input_dim:
                        return new
                    else:
                        return recursive_check(M, Att, i - 1)
                else:
                    return new
            
            new = recursive_check(M, Att, -1)

            Att = Att + tf.one_hot(new, input_dim, dtype = tf.int32) 
            NumberOfPoints = tf.reduce_prod(Att)
            return NumberOfPoints, Att

        NumberOfPoints, Att = tf.while_loop(lambda NumberOfPoints, Att: tf.less(NumberOfPoints, n), lambda NumberOfPoints, Att: foo(Att), [NumberOfPoints, Att])
                
        #print(out1)
        #print(out2)
        #grid = tf.meshgrid(*[I for i in range(k)])
        #print('what', tf.gather(idx[0,:], range(Att[0])))
        out = tf.meshgrid(*[tf.gather(idx[i,:], range(Att[i])) for i in range(input_dim)])
        #print(out)
        out = tf.concat([tf.reshape(out[i], (-1,1)) for i in range(input_dim)], axis = -1)
        #print(out)
        return out
    #itm = GetBiggestSubSum(BXnew[1,:,:])
    #print(itm)
    return tf.map_fn(lambda x: GetBiggestSubSum(x, n), BXnew, 
            fn_output_signature = tf.RaggedTensorSpec(shape = [None, input_dim], ragged_rank = 0, dtype = tf.int32)) 
def uniq(J):
    return np.unique(J, axis = 0)

def uniq_keep_order(J):
    J = np.array(J)
    return pd.DataFrame(J).drop_duplicates().values

if __name__ == '__main__':
    #n = 5; k = 6
    #out = GetAllPairs(n, k)
    #print(out)
    X = np.array([[0.0112, 0.01266, 0.00658, 0.536]])#, 
        #[0.91, 0.89, 0.59, 0.959],
        #[0.5, .5, .5, .5]])
    #I = GetNearGridPoints(X, 10)
    #I = GetNearGridPoints2(X, 10, 4, 100)
    #print(I)
    print(X)
    from gpmaniflow.surfaces import BernsteinPolynomial
    B = BernsteinPolynomial(10)
    BX = B(X)
    print(BX)
    I0 = GetNearGridPoints3(BX, 4, 10,  n = 2000)
    print(I0)
    print(I0.dtype)
    print(X)
    #print(I.shape)
    #print("onehot:", tf.one_hot(2, 4))
