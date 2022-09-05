import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from gpflow import logdensities
from gpflow.base import Parameter
from gpflow.models import BayesianModel
from gpflow.config import default_float
from gpflow.models.model import InputData, RegressionData
from gpflow.likelihoods import Likelihood
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin

from gpmaniflow.surfaces import BezierButtress, LogBezierButtress
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
            num_data = None,
            perm = None, # means random
            num_perm = 20
            ):
        super().__init__()
        self.input_dim = input_dim
        if isinstance(orders, int):
            listo = [orders] * input_dim
        self.orders = listo
        self.num_data = num_data # Important to set when using minibatches
        if perm is None:
            self.num_perm = num_perm
        else:
            self.num_perm = perm.shape[0]
        self.likelihood = likelihood
        
        self.BB = BezierButtress(input_dim = input_dim, orders = listo, perm = perm, num_perm = self.num_perm)        

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        X, Y = data
        assert X.shape[1] == self.input_dim
        kl = self.BB.kl()
        f_mean, f_var = self.predict_f(X)
        
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is None:
            scale = 1
        else:
            scale = tf.cast(self.num_data, var_exp.dtype) / tf.cast(tf.shape(X)[0], var_exp.dtype)
        
        return scale * tf.reduce_sum(var_exp) - kl 

    def predict_f(self, Xnew: InputData):       
        f_mean = self.BB.f_mean(Xnew)
        f_var = self.BB.f_var(Xnew)
        return f_mean, f_var

    def predict_y(self, Xnew: InputData):
        f_mean, f_var = self.predict_f(Xnew)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)


class LogBezierProcess(BayesianModel, ExternalDataTrainingLossMixin): 
    
    def __init__(
            self,
            input_dim,
            likelihood: Likelihood, # Testing phase.
            orders = 1,
            num_data = None,
            perm = None, # means random
            num_perm = 20
            ):
        super().__init__()
        self.input_dim = input_dim
        if isinstance(orders, int):
            listo = [orders] * input_dim
        self.orders = listo
        self.num_data = num_data # Important to set when using minibatches
        if perm is None:
            self.num_perm = num_perm
        else:
            self.num_perm = perm.shape[0]
        self.likelihood = likelihood
        
        self.BB = LogBezierButtress(input_dim = input_dim, orders = listo, perm = perm, num_perm = self.num_perm)        

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)
    
    def variational_expectations(self, f_mean, f_var, Y):
        # f_mean, f_var are [N,1]
        num_samples = 20
        S = tf.random.normal((tf.shape(f_mean)[0], num_samples), dtype = default_float())
        #print(S)
        mu = tf.math.log( f_mean / (1 + f_var/f_mean**2)**0.5 )
        sig = tf.math.log( 1 + f_var/f_mean**2 ) ** 0.5
        #print(mu)
        S = mu + sig*S
        #print(S)
        S = tf.math.exp(S) # LOGNORMALS
        out = logdensities.gaussian(Y, S, self.likelihood.variance)
        #print(out)
        out = tf.reduce_mean(out, axis = -1)
        #print(out)
        return out

    def elbo(self, data: RegressionData) -> tf.Tensor:
        X, Y = data
        assert X.shape[1] == self.input_dim
        kl = self.BB.kl()
        #kl = 0
        f_mean, f_var = self.predict_f(X)
        
        var_exp = self.variational_expectations(f_mean, f_var, Y)
        #print(var_exp)
        # CHANGE
        if self.num_data is None:
            scale = 1
        else:
            scale = tf.cast(self.num_data, var_exp.dtype) / tf.cast(tf.shape(X)[0], var_exp.dtype)
        
        return scale * tf.reduce_sum(var_exp) - kl 

    def predict_f(self, Xnew: InputData):       
        f_mean = self.BB.f_mean(Xnew)
        f_var = self.BB.f_var(Xnew)
        return f_mean, f_var

    def predict_y(self, Xnew: InputData):
        f_mean, f_var = self.predict_f(Xnew)
        # CHANGE
        #return self.likelihood.predict_mean_and_var(f_mean, f_var)
