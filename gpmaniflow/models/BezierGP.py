import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import Parameter
from gpflow.models import BayesianModel
from gpflow.config import default_float
from gpflow.models.model import InputData, RegressionData
from gpflow.likelihoods import Likelihood

from gpmaniflow.surfaces import BernsteinPolynomial, ControlPoints#, AmortisedControlPoints
from gpmaniflow.utils import GetAllListPairs

class BezierProcess(BayesianModel): # Where should we inherit from?
    '''
    Implementation of a Bezier Stochastic Process. 
    '''
    def __init__(
            self,
            input_dim,
            likelihood: Likelihood, # Testing phase.
            order = 1,
            output_dim =1,
            amortize = False
            ):
        super().__init__()
        self.input_dim = input_dim
        self.order = order
        self.output_dim = output_dim
        self.B = BernsteinPolynomial(self.order)
        if not amortize:
            self.P = ControlPoints(input_dim = self.input_dim, output_dim = self.output_dim, orders = self.order)
        else:
            #self.P = AmortisedControlPoints(input_dim = self.input_dim, orders = self.order)
            raise NotImplementedError
    
    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        #X, Y = data
        raise NotImplementedError

    def predict_f(self, Xnew: InputData):       
        assert Xnew.shape[1] == self.input_dim
        I = GetAllListPairs(self.order+1,self.input_dim) 
        outB = self.B(Xnew)
        outB = tf.gather(outB, I, axis = 2)
        outB = tf.linalg.diag_part(tf.transpose(outB, perm = (0,2,1,3) )) # [N, (order + 1)^d, d]
        outB = tf.reduce_prod(outB, axis = 2) # [N, (order + 1)^d]
        f_mean = tf.gather_nd(self.P.variational_posteriors.loc, I)
        f_var = tf.gather_nd(self.P.variational_posteriors.scale, I)
        f_mean = tf.matmul(outB, f_mean)
        f_var = tf.matmul(outB ** 2, f_var ** 2)
        return f_mean, f_var

if __name__ == '__main__':
    from gpflow.likelihoods import Gaussian
    m = BezierProcess(input_dim=3, order = 1, output_dim = 3, likelihood = Gaussian())
    from gpflow.utilities import print_summary
    print_summary(m)
    import numpy as np
    Xnew = np.array([[1., 1., 1.], [0.5, 0.5, 0.5], [0.0, 0., 0.]])
    print(Xnew)
    f_mean, f_var = m.predict_f(Xnew)
    print(f_mean)
    print(f_var)

