import numpy as np
import tensorflow as tf
 
def factorial(n):
    return np.prod(range(1,n+1))

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