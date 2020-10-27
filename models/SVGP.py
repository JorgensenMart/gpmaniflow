import tensorflow as tf
from gpflow.conditionals import conditional
from gpflow.models import SVGP
from gpflow.models.model import InputData
from gpmaniflow.kernels import dSquaredExponential

class SVGP(SVGP):
    ''' 
    Only works with prior mean constant zero.
    Only works for SquaredExponential kernel.
    '''
    def predict_df(self, Xnew: InputData, full_cov=False, full_output_cov=False):
        # Create derivative kernel
        derivative_kernel = dSquaredExponential(variance = self.kernel.variance,
                                                lengthscales = self.kernel.lengthscales)
        
        q_mu = self.q_mu 
        q_sqrt = self.q_sqrt 
        
        mu, var = conditional(
            Xnew, #[N,d]
            self.inducing_variable, #[M,d]
            derivative_kernel, # Throw in [Nd x D]
            q_mu, #[M,D]
            q_sqrt=q_sqrt, #WHAT HAPPENS HERE THEN?
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        #mu should be [Nd, D] same with var
        mu = tf.reshape(mu, [Xnew.shape[0], q_mu.shape[1], Xnew.shape[1]]) # [N, D, d]
        var = tf.reshape(var, mu.shape)
        return mu, var