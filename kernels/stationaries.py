import tensorflow as tf
import tensorflow.linalg.LinearOperatorKronecker as Kronecker
import gpflow
from gpflow.base import TensorLike
from gpflow.inducing_variables import InducingPoints
from gpflow.utilities import Dispatcher
from gpflow.utilities.ops import difference_matrix

class dSquaredExponential(gpflow.kernels.SquaredExponential):    
    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))
        
    def K(self, X1, X2 = None):
        dK = Dispatcher("dK")
        if X2 == None:
            @dK.register(InducingPoints, object)
            def _dK(X1,X2):
                r2 = self.scaled_squared_euclid_dist(X1.Z)
                return self.K_r2(r2) # [M, M]
            
            @dK.register(TensorLike, object)
            def _dK(X1,X2):
                N, d = X1.shape
                d_by_d = tf.ones([d, d], dtype = default_float())
                r2 = self.scaled_squared_euclid_dist(X1.Z)
                
                diff = difference_matrix(self.scale(X1),self.scale(X1)) # [N, N, d]
                diff = tf.expand_dims(diff, axis = -1) # [N, N, d, 1]
                diff = tf.matmul(diff, -diff, transpose_b = True) # [N, N, d, d] # I THINK ONE DIFF SHOULD BE NEGATIVE
                diff = tf.reshape(diff, [N*d, N*d]) # [Nd, Nd]
                                
                out = Kronecker(d_by_d, self.K_r2(r2)) * diff #[Nd,Nd]
                return out # [Nd, Nd] (d is input dimensionality)
        else:
            @dK.register(InducingPoints, TensorLike)
            def _dK(Z, X):
                N, d = X.shape
                M = Z.num_inducing
                one_by_d = tf.ones([1, d], dtype = default_float())
                r2 = self.scaled_squared_euclid_dist(Z.Z, X) # [M, N]
                
                diff = difference_matrix(self.scale(Z.Z),self.scale(X)) # [M, N, d]
                diff = tf.reshape(-diff, [M, N*d]) # [M, Nd]
                                
                out = Kronecker(one_by_d, self.K_r2(r2)) * diff #[M,Nd] 
                return out #[M, Nd]
            
        return dK(X1,X2)