import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from gpflow.config import default_float
from gpflow.models.model import InputData
from gpflow.base import Parameter, Module
from gpflow.utilities import positive

from gpmaniflow.utils import binomial_coef as binom

class ControlPoints(Module):
    def __init__(self, input_dim, output_dim = 1, orders = 1):
        self.num_points = (orders + 1) ** input_dim # What are we doing about potential overflow?
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.orders = orders

        self.priors = tfp.distributions.Normal(
                loc = tf.zeros([self.orders + 1] * self.input_dim + [self.output_dim], dtype = default_float()),
                scale = tf.ones([self.orders + 1] * self.input_dim + [self.output_dim], dtype = default_float())
                )
        
        self.variational_loc = Parameter(value = tf.zeros([self.orders + 1] * self.input_dim + [self.output_dim], dtype = default_float() )) 
        self.variational_scale = Parameter(value = tf.ones([self.orders + 1] * self.input_dim + [self.output_dim], dtype = default_float() ), transform = positive()) 
        
        self.variational_posteriors = tfp.distributions.Normal(loc = self.variational_loc, scale = self.variational_scale) 

    def kl_divergence(self):
        return tfp.distributions.kl_divergence(self.variational_posteriors, self.priors)

class AmortisedControlPoints(Module):
    def __init__(self, input_dim, output_dim = 1, orders = 1, nn = None, batch_size = None):
        self.num_points = (orders + 1) ** input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.orders = orders
        
        if nn == None:
            self.nn = vanilla_mlp(self.input_dim, self.output_dim) # This should be implemented for several output dims
        else:
            self.nn = nn
        self.nn.build()
        if batch_size is None:
            self.batch_size = min(2000, self.num_points)
        else:
            self.batch_size = batch_size
    
    def forward(self, mini_batch):
        return self.nn(mini_batch) # [B, P, n_params]   
    
    def kl_divergence(self, mini_batch):
        amP = self.nn(mini_batch)
        var_approx = tfp.distributions.Normal(loc = amP[:,:,0], scale = tf.exp(amP[:,:,1]))
        out_shape = tf.shape(var_approx.loc)
        return tfp.distributions.kl_divergence(
                var_approx,
                tfp.distributions.Normal(loc = tf.zeros(out_shape, dtype = default_float()), scale = tf.ones(out_shape, dtype = default_float())))

    #pass


class BernsteinPolynomial():
    def __init__(self, orders = None):
        self.orders = orders
    
    def __call__(self, Xnew: InputData):
        return self.forward(Xnew)

    def forward(self, Xnew: InputData, mini_batch = None):
        if mini_batch is not None:
            raise NotImplementedError
        else:
            O = self.orders + 1
            Xnew = tf.stack([Xnew] * O, axis = 2)
            X1 = tf.math.pow(Xnew, range(O))
            X2 = tf.math.pow(1. - Xnew, np.array(self.orders) - range(O))
            B_out = X1 * X2 * binom(np.array(self.orders), range(O))
            return B_out

def vanilla_mlp(input_dim, output_dim):
    mlp = tf.keras.Sequential([
            tf.keras.Input(shape = (input_dim,)),
            tf.keras.layers.Dense(input_dim),
            tf.keras.layers.Dense(30, activation = "relu"),
            tf.keras.layers.Dense(30, activation = "relu"),
            #tf.keras.layers.Dense(30, activation = "relu"),
            #tf.keras.layers.Dense(48, activation = "relu"), 
            tf.keras.layers.Dense(2 * output_dim),
            tf.keras.layers.Reshape((output_dim, 2)),
            tf.keras.layers.Lambda(lambda x: tf.cast(x,  default_float()))
            ])
    return mlp

if __name__ == '__main__':
    #P = AmortisedControlPoints(input_dim = 1)
    #print(P.nn)
    #out = P.nn(np.array([[1]]))
    #print(out)
    B = BernsteinPolynomial(orders = 10)
    b_out = B(tf.constant([[0.6, 0.5, 0.4], [0.6, 0.5, 0.4]], dtype = default_float() ))
    print(b_out)
    print(tf.reduce_sum(b_out, axis = 2))
#if __name__ == '__main__':
    
#    B = BernsteinPolynomial(orders = 5)
#    X = tf.constant([[0], [1]], dtype = default_float())
#    #print(X)
#    B_out = B(X)
#    print(B_out)
    
 #   I = tf.range(0,B.orders+1)
  #  print(I)
   # grid1 = tf.meshgrid(I)
    #print(grid1)
    #print(grid2)
    #print(grid3)
#    out = tf.concat( [tf.reshape(grid1, (-1, 1)), tf.reshape(grid2, (-1,1)), tf.reshape(grid3, (-1,1))], axis = -1)
#    print(out)

    # THIS GIVES ALL TRIPLETS, AND SHOULD EXTEND TO HIGHER: QUADRUPLETS ETC.
    #Is = tf.reshape( [0, 1, 2] * 64, (64, 3))
    #print(Is)
    #Bii = tf.gather_nd(B_out, [[[0,0]]])
    #print(Bii
    #myB = tf.gather(B_out, out, axis = 2)
    #print(tf.gather(B_out, out, axis = 2))
    #print(tf.linalg.diag_part(tf.transpose(myB, perm = (0,2,1,3))))
    
    #P = ControlPoints(input_dim = 3, orders = 3)
    #print(tf.gather_nd(P.variational_posteriors.loc, out))
    #print(P.priors)
    #print(tf.reduce_sum(P.kl_divergence()))

    # Now we're playing
    #N = 2; D = 3; O = 3
    #X = tf.zeros([N, D])
    #X = tf.stack([X] * O, axis = 2)
    #T = range(0,O)
#X = tf.math.pow(X, T)
 #   print(X)
    #import numpy as np
    #print(binom(np.array(5),np.array( [0, 1])))
        # TODO: make it work for multidimensional outputs
        # TODO: make it work for multidimensional outputs
