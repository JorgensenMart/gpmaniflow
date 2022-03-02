import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from gpflow.base import Parameter
from gpflow.models import BayesianModel
from gpflow.config import default_float
from gpflow.models.model import InputData, RegressionData
from gpflow.likelihoods import Likelihood
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin

from gpmaniflow.surfaces import BernsteinNetwork
from gpmaniflow.utils import GetAllListPairs

class BezierProcess(BayesianModel, ExternalDataTrainingLossMixin): 
    '''
    Implementation of a Bezier Stochastic Process. 
    '''
    def __init__(
            self,
            input_dim,
            likelihood: Likelihood, # Testing phase.
            orders = 1,
            num_data = None
            ):
        super().__init__()
        self.input_dim = input_dim
        if isinstance(orders, int):
            print("hey")
            listo = [orders] * input_dim
        self.orders = listo
        print(listo)
        self.num_data = num_data # Important to set when using minibatches
        self.likelihood = likelihood
        
        self.BN = BernsteinNetwork(input_dim = input_dim, orders = listo)        

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        X, Y = data
        #kl = self.P.num_points * tf.reduce_mean(self.P.kl_divergence(global_I))
        #print('KL:', kl)
        kl = 0
        f_mean, f_var = self.predict_f(X)
        
        #var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        var_exp = self.likelihood.predict_log_density(f_mean, f_var, Y) 
        if self.num_data is None:
            scale = 1
        else:
            scale = tf.cast(self.num_data, var_exp.dtype) / tf.cast(tf.shape(X)[0], var_exp.dtype)
        
        return scale * tf.reduce_sum(var_exp) - kl 

    def predict_f(self, Xnew: InputData, I = None, pi = None, approx = False, outB = None):       
        f_mean = self.BN.f_mean(Xnew)
        f_var = self.BN.f_var(Xnew)
        return f_mean, f_var
