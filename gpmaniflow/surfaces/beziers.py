import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.config import default_float
from gpflow.models.model import InputData
from gpflow.base import Parameter
from gpflow.utilities import positive

from gpmaniflow.utils import binomial_coef as binom

class ControlPoints():
    def __init__(self, input_dim, output_dim = 1, orders = 1):
        self.num_points = (orders + 1) ** input_dim # What are we doing about potential overflow?
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.orders = orders

        self.priors = tfp.distributions.Normal(
                loc = tf.zeros([self.orders + 1] * self.input_dim + [self.output_dim], dtype = default_float()),
                scale = tf.ones([self.orders + 1] * self.input_dim + [self.output_dim], dtype = default_float())
                )
        # TODO: make it work for multidimensional outputs
        
        self.variational_posteriors = tfp.distributions.Normal(
                loc = Parameter(value = tf.zeros([self.orders + 1] * self.input_dim + [self.output_dim], dtype = default_float() )), 
                scale = Parameter(value = tf.ones([self.orders + 1] * self.input_dim + [self.output_dim], dtype = default_float() ), transform = positive())
                )

    def kl_divergence(self):
        return tfp.distributions.kl_divergence(self.variational_posteriors, self.priors)

class AmortisedControlPoints(ControlPoints):
    pass


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
            B_out = tf.zeros([Xnew.shape[0],Xnew.shape[1],O], dtype = default_float())
            B_out = tf.Variable(B_out) # TensorFlow only can assign on variable.
            for i in range(O): #TODO: Avoid this loop.
                B_out[...,i].assign( Xnew ** i * (1-Xnew) ** (self.orders-i) * binom(self.orders, i) )
            B_out = tf.convert_to_tensor(B_out) # Not a Variable anymore.
            return B_out

if __name__ == '__main__':
    
    #B = BernsteinPolynomial(orders = 5)
    #X = tf.constant([[0], [1]], dtype = default_float())
    #print(X)
    #B_out = B(X)
    #print(B_out)
    
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
    
    P = ControlPoints(input_dim = 3, orders = 3)
    #print(tf.gather_nd(P.variational_posteriors.loc, out))
    print(P.priors)
    #print(tf.reduce_sum(P.kl_divergence()))
