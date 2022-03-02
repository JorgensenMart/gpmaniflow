import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from gpflow.config import default_float
from gpflow.models.model import InputData
from gpflow.base import Parameter, Module
from gpflow.utilities import positive

from gpmaniflow.utils import binomial_coef as binom
from gpmaniflow.utils import GetAllListPairs

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
        # test
        self.prior_scale = Parameter(value = 1, dtype = default_float(), transform = positive())
        self.prior_adjustment = prior_adjusting(self.orders) # [o + 1, 1]

        if nn == None:
            self.nn = vanilla_mlp(self.input_dim, self.output_dim) # This should be implemented for several output dims
        else:
            self.nn = nn
        self.nn.build()
        if batch_size is None:
            self.batch_size = min(2000, self.num_points)
        else:
            self.batch_size = batch_size
    
    def prior_sc(self, mini_batch):
        sc = tf.gather(self.prior_adjustment, mini_batch, axis = 0) # [B, d, 1]
        sc = tf.reduce_prod(sc, axis = 1) # [B, 1]
        return sc
    
    def forward(self, mini_batch):
        return self.nn(mini_batch) # [B, P, n_params]   
    
    def kl_divergence(self, mini_batch):
        amP = self.nn(mini_batch / self.orders)
        #amP = self.nn(mini_batch / self.orders)
        var_approx = tfp.distributions.Normal(loc = amP[:,:,0], scale = tf.math.exp(amP[:,:,1]))
        out_shape = tf.shape(var_approx.loc)
        return tfp.distributions.kl_divergence(
                var_approx,
                #tfp.distributions.Normal(loc = tf.zeros(out_shape, dtype = default_float()), scale = tf.ones(out_shape, dtype = default_float())))
                tfp.distributions.Normal(loc = tf.zeros(out_shape, dtype = default_float()), scale = self.prior_scale *  self.prior_sc(tf.cast(mini_batch, dtype = tf.int32))))

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


class BernsteinNetwork(tf.keras.layers.Layer):
    def __init__(self, input_dim = 1, orders = 1):
        super(BernsteinNetwork,self).__init__()
        self.orders = orders
        print(orders)
        _dummy = [0]; _dummy.extend(orders)
        print(_dummy)
        self.input_dim = input_dim
        
        w_init = tf.random_normal_initializer()
        self.meanw = (*[tf.Variable(initial_value = w_init(shape = (_dummy[i] + 1, _dummy[i+1] + 1), dtype = default_float()), trainable = True,) for i in range(input_dim)],)
        self.varw = (*[tf.Variable(initial_value = w_init(shape = (_dummy[i] + 1, _dummy[i+1] + 1), dtype = default_float()), trainable = True,) for i in range(input_dim)],)
        
        self.B = BernsteinPolynomial(10)
        self.prior_sc = prior_adjusting(10)
        print(self.prior_sc)

    def f_mean(self, Xnew):
        outB = self.B(Xnew) # [N, d, o + 1]
        f = tf.ones((tf.shape(outB)[0], 1), dtype = default_float()) # [N, 1]
        for i in range(self.input_dim):
            f = tf.matmul(f, self.meanw[i]) # [N, o+1]
            #print(f)
            f = f * outB[:,i,:] # [N, o + 1]
        f = tf.reduce_sum(f, axis = 1, keepdims = True)
        return f# [N, 1]

    def f_var(self, Xnew):
        outB = self.B(Xnew) # [N, d, o + 1]
        f = tf.ones((tf.shape(outB)[0], 1), dtype = default_float()) # [N, 1]
        for i in range(self.input_dim):
            f = tf.matmul(f, tf.math.exp(self.varw[i]) * tf.transpose(self.prior_sc ** 2)) # [N, o+1]
            f = f * outB[:,i,:] ** 2 # [N, o + 1]
        f = tf.reduce_sum(f, axis = 1, keepdims = True)
        return f# [N, 1]

    def P_mean(self, batch):
        # batch is [N, d]
        P = tf.ones((tf.shape(batch)[0],1), dtype = default_float())
        for i in range(self.input_dim):
            P = tf.matmul(P, self.meanw[i]) 
            #print(tf.one_hot(batch[:,i] * self.orders[i], self.orders[i] + 1, dtype = default_float()))
            P = P * tf.one_hot(batch[:,i] * self.orders[i], self.orders[i] + 1, dtype = default_float())
        P = tf.reduce_sum(P, axis = 1, keepdims = True)
        return P# [N, 1}

    def P_var(self, batch):
        P = tf.ones((tf.shape(batch)[0],1), dtype = default_float())
        for i in range(self.input_dim):
            P = tf.matmul(P, tf.math.exp(self.varw[i])) 
            #print(tf.one_hot(batch[:,i] * self.orders[i], self.orders[i] + 1, dtype = default_float()))
            P = P * tf.one_hot(batch[:,i] * self.orders[i], self.orders[i] + 1, dtype = default_float())
        P = tf.reduce_sum(P, axis = 1, keepdims = True)
        return P # [N, 1]

def prior_adjusting(order):
    if order > 25:
        raise NotImplementedError

    I = tf.cast(GetAllListPairs(order + 1, 1), default_float()) / order
    B = BernsteinPolynomial(order)
    BX = B(I)
    P = tf.linalg.inv( tf.squeeze(BX ** 2, axis = 1) )
    return tf.sqrt( tf.matmul(P, tf.ones([order + 1, 1], dtype = default_float())) ) 





def vanilla_mlp(input_dim, output_dim):
    mlp = tf.keras.Sequential([
            tf.keras.Input(shape = (input_dim,)),
            tf.keras.layers.Dense(input_dim),
            #tf.keras.layers.Dense(40, activation = "relu"),
            tf.keras.layers.Dense(40, activation = "relu"),
            tf.keras.layers.Dense(40, activation = "relu"),
            #tf.keras.layers.Dense(48, activation = "elu"), 
            tf.keras.layers.Dense(2 * output_dim),
            tf.keras.layers.Reshape((output_dim, 2)),
            tf.keras.layers.Lambda(lambda x: tf.cast(x,  default_float()))
            ])
    return mlp

if __name__ == '__main__':
    l = [1]; j = [2, 3]
    l.extend(j)
    print(l)
    N = BernsteinNetwork(input_dim = 2, orders = [3,3])
    print(N.w)
    f = N.f_mean(tf.constant([[1., 1.]], dtype = default_float()))
    print(f)
    #P = AmortisedControlPoints(input_dim = 1)
    #print(P.nn)
    #out = P.nn(np.array([[1]]))
    #print(out)
    #B = BernsteinPolynomial(orders = 10)
    #b_out = B(tf.constant([[0.05, 0.05, 0.05], [0.3, 0.9, 0.4]], dtype = default_float() ))
    #print(b_out) # [N, d, B]
    #I = tf.constant([[[0, 0, 0]], [[1, 1, 1]]])
    #print(I.shape)
    #print(tf.gather(b_out,I, axis = 2, batch_dims = 1))
    #b_out = tf.gather(b_out,I, axis = 2, batch_dims = 1)
    #print(tf.reduce_sum(b_out, axis = 2))
    
    #P = prior_adjusting(10)
    #print(P)
    #I = GetAllListPairs(3, 2)
    #print(I)
    #pout = tf.gather(P, I, axis = 0)
    #print(pout)
    #pout = tf.reduce_prod(pout, axis = 1)
    #print(pout)
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
