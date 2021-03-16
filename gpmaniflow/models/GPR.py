import tensorflow as tf
from typing import Optional

from gpflow.models import GPR
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import InputData, RegressionData
from gpflow.inducing_variables import InducingPoints
from gpmaniflow.kernels import dSquaredExponential
from gpmaniflow.conditionals import conditional
from gpmaniflow.samplers import sample_matheron

class GPR(GPR):
    ''' 
    Only works with prior mean constant zero.
    Only works for SquaredExponential kernel.
    '''
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        _, Y_data = data
        super().__init__(data, kernel, mean_function, noise_variance = noise_variance)
        self._MatheronSampler = None
        
    def predict_df(self, Xnew: InputData, full_cov=False, full_output_cov=False):
        # Create derivative kernel
        derivative_kernel = dSquaredExponential(variance = self.kernel.variance,
                                                lengthscales = self.kernel.lengthscales, eager_execution=False)
         
        X, Y = self.data
        
        mu, var = conditional(
            Xnew, #[N,d]
            InducingPoints(X), #Passing through InducingPoints to get correct Kuu etc.
            derivative_kernel, 
            Y,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
        )
        #mu should be [Nd, D] same with var
        mu = tf.reshape(mu, [Xnew.shape[0], Y.shape[1], Xnew.shape[1]]) # [N, D, d]
        var = tf.reshape(var, mu.shape) # Here should change if full_cov is True
        return mu, var
    
    @property
    def MatheronSampler(self):
        return self._MatheronSampler
    
    @MatheronSampler.setter
    def MatheronSampler(self, initializer):
        from_df, num_samples, num_basis = initializer
        if from_df:
            X, Y = self.data
            derivative_kernel = dSquaredExponential(variance = self.kernel.variance,
                                                lengthscales = self.kernel.lengthscales, eager_execution=False)
            
            self._MatheronSampler = sample_matheron(InducingPoints(X), derivative_kernel, Y, likelihood_var = self.likelihood.variance,
                                                    num_samples = num_samples, num_basis = num_basis)
        else:
            X, Y = self.data
            self._MatheronSampler = sample_matheron(InducingPoints(X), self.kernel, Y, likelihood_var = self.likelihood.variance,
                                                    num_samples = num_samples, num_basis = num_basis)