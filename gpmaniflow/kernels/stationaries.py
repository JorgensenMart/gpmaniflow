import tensorflow as tf
#from tensorflow.linalg import LinearOperatorKronecker as Kronecker

import gpflow
from gpflow.config import default_float
from gpflow.base import TensorLike
from gpflow.inducing_variables import InducingPoints
from gpflow.utilities import Dispatcher
from gpflow.utilities.ops import difference_matrix

class dSquaredExponential(gpflow.kernels.SquaredExponential):
    
    def Kronecker(self, A, B):
        shape = tf.stack([tf.shape(A)[0] * tf.shape(B)[0], tf.shape(A)[1] * tf.shape(B)[1]])
        return tf.reshape(tf.expand_dims(tf.expand_dims(A, 1), 3) * tf.expand_dims(tf.expand_dims(B, 0), 2), shape)
    
    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X)) # This should be more efficient
    
    dK = Dispatcher("dK")
    
    @dK.register(object,InducingPoints, InducingPoints)
    def _dK(self,X1,X2):
        r2 = self.scaled_squared_euclid_dist(X1.Z)
        return self.K_r2(r2) # [M, M]
            
    @dK.register(object,TensorLike, TensorLike)
    def _dK(self,X1,X2):
        N, d = X1.shape
        d_by_d = tf.ones([d, d], dtype = default_float())
        r2 = self.scaled_squared_euclid_dist(X1)
        
        diff = difference_matrix(self.scale_sq(X1),self.scale_sq(X1)) # [N, N, d]
        diff = tf.expand_dims(diff, axis = -1) # [N, N, d, 1]
        diff = tf.matmul(diff, -diff, transpose_b = True) # [N, N, d, d] # I THINK ONE DIFF SHOULD BE NEGATIVE
        diff = tf.reshape(diff, [N*d, N*d]) # [Nd, Nd]

        out = self.Kronecker(d_by_d, self.K_r2(r2)) * diff #[Nd,Nd]
        return out # [Nd, Nd] (d is input dimensionality)
    
    @dK.register(object, InducingPoints, TensorLike)
    def _dK(self, Z, X):
        N, d = X.shape
        M = Z.num_inducing
        one_by_d = tf.ones([1, d], dtype = default_float())
        r2 = self.scaled_squared_euclid_dist(Z.Z, X) # [M, N]

        diff = difference_matrix(self.scale_sq(Z.Z),self.scale_sq(X)) # [M, N, d]
        diff = tf.reshape(diff, [M, N*d]) # [M, Nd]

        out = self.Kronecker(one_by_d, self.K_r2(r2)) * diff #[M,Nd] 
        return out #[M, Nd]
    
    def K(self, X1, X2 = None):
        if X2 is None:
            X2 = X1
        return self.dK(self,X1,X2)
    
    def scale_sq(self, X):
        X_scaled = X / self.lengthscales**2 if X is not None else X
        return X_scaled