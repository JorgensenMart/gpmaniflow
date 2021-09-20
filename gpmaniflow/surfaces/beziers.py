import tensorflow as tf
from gpmaniflow.utils import binomial_coef as binom
from gpflow.config import default_float
from gpflow.models.model import InputData

class BezierSurface():
    def __init__(self, intervals = None, input_dim = None, orders = None):
        self.orders = orders # Should default to ones
#        self.dimension =

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
            for i in range(O):
                B_out[...,i].assign( Xnew ** i * (1-Xnew) ** (O-i) * binom(O, i) )
            B_out = tf.convert_to_tensor(B_out) # Not a Variable anymore.
            return B_out

if __name__ == '__main__':
    B = BernsteinPolynomial(orders = 3)
    X = tf.constant([[0.3, 0.5], [0.9, 0.1]], dtype = default_float())
    print(X)
    B_out = B(X)
    print(B_out)
