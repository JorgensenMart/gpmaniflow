import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from gpflow.config import default_float
from gpflow.models.model import InputData
from gpflow.base import Parameter, Module
from gpflow.utilities import positive

from gpmaniflow.utils import binomial_coef as binom
from gpmaniflow.utils import GetAllListPairs

class BernsteinPolynomial():
    def __init__(self, orders = None):
        self.orders = orders
    
    def __call__(self, Xnew: InputData):
        return self.forward(Xnew)

    def forward(self, Xnew: InputData):
        O = self.orders + 1
        Xnew = tf.stack([Xnew] * O, axis = 2)
        X1 = tf.math.pow(Xnew, range(O))
        X2 = tf.math.pow(1. - Xnew, np.array(self.orders) - range(O))
        B_out = X1 * X2 * binom(np.array(self.orders), range(O))
        return B_out


class BezierButtress(tf.keras.layers.Layer):
    def __init__(self, input_dim = 1, orders = 1, perm = None, num_perm = 20):
        super(BezierButtress,self).__init__()
        self.orders = orders
        _dummy = [0]; _dummy.extend(orders)
        self.input_dim = input_dim
        self.num_perm = num_perm
        
        meanw_init = tf.random_normal_initializer(mean = (0.0001/ self.num_perm) ** (1/self.input_dim), stddev = 0.1)
        #meanw_init = tf.zeros_initializer()
        
        varw_init = tf.random_normal_initializer(mean = -0.1, stddev = 0.1)
        #varw_init = tf.zeros_initializer()
        if perm is None:
            _perm = np.tile(range(self.input_dim), (self.num_perm,1))
            self.perm = np.apply_along_axis(np.random.permutation, axis=1, arr=_perm)
        else:
            self.perm = perm
        self.meanw = (*[tf.Variable(initial_value = meanw_init(shape = (self.num_perm, _dummy[i] + 1, _dummy[i+1] + 1), dtype = default_float()), trainable = True,) for i in range(input_dim)],)
        self.varw = (*[tf.Variable(initial_value = varw_init(shape = (self.num_perm, _dummy[i] + 1, _dummy[i+1] + 1), dtype = default_float()), trainable = True,) for i in range(input_dim)],)
        
        self.B = BernsteinPolynomial(self.orders[0])
        #self.prior_sc = tf.ones(self.orders[0] + 1, default_float())#prior_adjusting(self.orders[0])
        self.prior_sc = prior_adjusting(self.orders[0])
        self.prior_variance = Parameter(value = [1 / (self.num_perm)] * self.num_perm, dtype = default_float(), trainable = True, transform = positive())
        self.posterior_precision = Parameter(value = [self.num_perm] * self.num_perm, dtype = default_float(), trainable = True, transform = positive()) 
    
    def f_mean(self, Xnew):
        outB = self.B(Xnew) # [N, d, o + 1]
        outB = tf.gather(outB, self.perm, axis = 1)
        outB = tf.transpose(outB, (1,0,2,3)) # [R, N, d, o + 1]
        f = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1] 
        for i in range(self.input_dim):
            f = tf.matmul(f, self.meanw[i]) # [R, N, o+1]
            f = f * outB[:,:,i,:] # [R, N, o + 1]
        f = tf.reduce_sum(f, axis = -1, keepdims = True)
        f = tf.reduce_sum(f, axis = 0)
        return f# [N, 1]

    def f_var(self, Xnew):
        outB = self.B(Xnew) # [N, d, o + 1]
        outB = tf.gather(outB, self.perm, axis = 1)
        outB = tf.transpose(outB, (1,0,2,3)) # [R, N, d, o + 1]
        f = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1]
        for i in range(self.input_dim):
            f = tf.matmul(f, tf.math.exp(self.varw[i]) * tf.transpose(self.prior_sc ** 2)) # [R, N, o+1]
            f = f * outB[:,:,i,:] ** 2 # [R, N, o + 1]
        f = tf.reduce_sum(f, axis = -1, keepdims = True) # [R, N, 1]
        f = tf.reduce_sum(f / self.posterior_precision[:,None,None], axis = 0)
        return f # [N, 1]

    def kl(self):
        num_points = tf.reduce_prod(tf.cast(self.orders, dtype = default_float()) + 1.0)
        meanterm = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        traceterm = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        detterm = 0.0 
        for i in range(self.input_dim):
            meanterm = tf.matmul(meanterm, self.meanw[i] ** 2 * tf.transpose(1 / self.prior_sc ** 2)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            traceterm = tf.matmul(traceterm, tf.math.exp(self.varw[i])) / (tf.cast(self.orders[i], default_float()) + 1.0)
            detterm += tf.reduce_mean(self.varw[i]) * (self.num_perm)
        meanterm = tf.reduce_sum(meanterm, axis = (1,2))
        traceterm = tf.reduce_sum(traceterm, axis = (1,2))
        meanterm = tf.reduce_sum(meanterm / self.prior_variance)
        traceterm = tf.reduce_sum(traceterm / (self.prior_variance * self.posterior_precision))
        
        return 0.5 * (traceterm + meanterm - 1.0 + tf.reduce_sum(tf.math.log(self.prior_variance * self.posterior_precision)) - detterm)

    def P_mean(self, batch):
        P = tf.ones((self.num_perm, tf.shape(batch)[0],1), dtype = default_float()) # [R, N, 1]
        for i in range(self.input_dim):
            P = tf.matmul(P, self.meanw[i]) # [R, N, o + 1]
            K = tf.one_hot(batch[:,i], self.orders[i] + 1, dtype = default_float())
            K = tf.tile(K[None, :, :], [self.num_perm, 1, 1])
            P = P * K
        P = tf.reduce_sum(P, axis = -1, keepdims = True)
        P = tf.reduce_sum(P, axis = 0)
        return P# [N, 1}

    def P_var(self, batch):
        P = tf.ones((self.num_perm, tf.shape(batch)[0],1), dtype = default_float()) # [R, N, 1]
        for i in range(self.input_dim):
            P = tf.matmul(P, tf.math.exp(self.varw[i]) * tf.transpose(self.prior_sc ** 2)) 
            K = tf.one_hot(batch[:,i], self.orders[i] + 1, dtype = default_float())
            K = tf.tile(K[None, :, :], [self.num_perm, 1, 1])
            P = P * K
        P = tf.reduce_sum(P, axis = -1, keepdims = True)
        P = tf.reduce_sum(P, axis = 0)
        return P # [N, 1]

    def integral(self):
        num_points = tf.reduce_prod(tf.cast(self.orders, dtype = default_float()) + 1) 
        f = tf.ones((self.num_perm, 1, 1), dtype = default_float()) # [R, N, 1] 
        for i in range(self.input_dim):
            f = tf.matmul(f, self.meanw[i]) # [R, N, o+1]
        f = tf.reduce_sum(f)
        return f / num_points

def prior_adjusting(order):
    if order > 25:
        raise NotImplementedError
    I = tf.cast(GetAllListPairs(order + 1, 1), default_float()) / order
    B = BernsteinPolynomial(order)
    BX = B(I)
    P = tf.linalg.inv( tf.squeeze(BX ** 2, axis = 1) )
    return tf.sqrt( tf.matmul(P, tf.ones([order + 1, 1], dtype = default_float()) )) 



if __name__ == '__main__':
    pass
