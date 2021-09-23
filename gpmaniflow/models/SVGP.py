import tensorflow as tf
from gpflow.models import SVGP
from gpflow.models.model import InputData

from gpmaniflow.kernels import dSquaredExponential
from gpmaniflow.conditionals import conditional
from gpmaniflow.samplers import sample_matheron

class SVGP(SVGP):
    ''' 
    Only works with prior mean constant zero.
    Only works for SquaredExponential kernel.
    '''
    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        num_data=None,
        MatheronSampler = None
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        super().__init__(
            kernel,
            likelihood,
            inducing_variable,
            num_latent_gps=num_latent_gps,
            whiten=whiten,
            num_data=num_data,
            mean_function=mean_function,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            q_diag=q_diag,
        )
        self._MatheronSampler = None
        
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
        var = tf.reshape(var, mu.shape) # Here should change if full_cov is True
        return mu, var
    
    @property
    def MatheronSampler(self):
        return self._MatheronSampler
    
    @MatheronSampler.setter
    def MatheronSampler(self, initializer):
        from_df, num_samples, num_basis = initializer
        if from_df:
            derivative_kernel = dSquaredExponential(variance = self.kernel.variance,
                                                lengthscales = self.kernel.lengthscales)
            self._MatheronSampler = sample_matheron(self.inducing_variable, derivative_kernel, self.q_mu, q_sqrt = self.q_sqrt, num_samples = num_samples, num_basis = num_basis)
        else:
            self._MatheronSampler = sample_matheron(self.inducing_variable, self.kernel, self.q_mu, q_sqrt = self.q_sqrt, num_samples = num_samples, num_basis = num_basis)
        
