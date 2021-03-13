import tensorflow as tf
import numpy as np
from typing import Optional

from gpflow.models.training_mixins import InputData, OutputData, InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from gpflow.kernels import Kernel, SquaredExponential
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.base import Parameter
from gpflow.utilities.ops import pca_reduce, square_distance

from gpmaniflow.models import SVGP, GPR
from gpmaniflow.kernels import dSquaredExponential
from gpmaniflow.likelihoods import Nakagami
from gpmaniflow.conditionals import conditional
from gpmaniflow.samplers import sample_matheron, initialize_sampler
from gpmaniflow.curves import BezierCurve

class IsoGPLVM(GPR):
    def __init__(
        self,
        data: OutputData,
        latent_dim: int,
        proximity_data: Optional[tf.Tensor] = None, # The non-Bayesian Iso-GPLVM >>needs<< the embedding, proximity is optional.
        X_data_mean: Optional[tf.Tensor] = None,
        kernel: Optional[Kernel] = None,
        mean_function: Optional[MeanFunction] = None,
    ):
        """
        Initialise IsoGPLVM object. 
        :param data: y data matrix, size N (number of points) x D (dimensions)
        :param latent_dim: the number of latent dimensions (Q)
        :param proximity_data: proximity matrix, size NxN
        :param X_data_mean: latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param mean_function: mean function, by default None.
        """
        if X_data_mean is None:
            X_data_mean = pca_reduce(data, latent_dim)
        
        if proximity_data is None:
            """
            If proximity data is not given, the Euclidean distances of the embeddings are used.
            """
            proximity_data = tf.sqrt(square_distance(data,data))
            
        num_latent_gps = X_data_mean.shape[1]
        if num_latent_gps != latent_dim:
            msg = "Passed in number of latent {0} does not match initial X {1}."
            raise ValueError(msg.format(latent_dim, num_latent_gps))

        if mean_function is None:
            mean_function = Zero()

        if kernel is None:
            kernel = SquaredExponential(lengthscales=tf.ones((latent_dim,)))

        if data.shape[1] < num_latent_gps:
            raise ValueError("More latent dimensions than observed.")

        gpr_data = (Parameter(X_data_mean), data_input_to_tensor(data))
        super().__init__(gpr_data, kernel, mean_function=mean_function)
    
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()
    
    def log_marginal_likelihood(self) -> tf.Tensor:
        X = self.data[1]
        # Sample draws from the Jacobian using Matheron
        self.MatheronSampler = initialize_sampler(from_df = True, num_samples = 50) # Initialize a Matheron sampler.
        # Reshape data from [N,d] -> [N*(N-1)/2,2,1,d] # All pairwise points
        I = range(len(X))
        out1, out2 = tf.meshgrid(I,tf.transpose(I))
        out = tf.concat([tf.reshape(out2,(-1,1)),tf.reshape(out1,(-1,1))],axis=-1)
        out = out[out[:,0] < out[:, 1]]
        pX = tf.expand_dims(tf.gather(X, out), 2) #All pairwise points
        # Compute curve-lengths
        def curve_length(x1_and_x2):
            C = BezierCurve(x1_and_x2)
            t = np.linspace(0,1, 20) # Consider throwing away the last index!
            J = self.MatheronSampler(C(t)) # [S, t, d, D]
            J = tf.transpose(J, perm = [0,1,3,2]) #[S, t, D, d]
            res = J @ tf.expand_dims(C.deriv(t), axis= 2) # [S, t, D, 1]
            res = tf.norm(res, axis = [-2,-1]) # [S, t]
            res = tf.reduce_mean(res, axis = 1)
            return res # S
        curve_lengths = tf.map_fn(pX, curve_length) # [N*(N-1)/2, S]
        # Estimate relevant Nakgami parameters
        Omega = tf.reduce_mean(tf.square(curve_lengths), axis = 1) #[N*(N-1)/2]
        m = tf.square(Omega) / (tf.reduce_mean(tf.square(tf.square(curve_lengths)), axis = 1) - Omega ** 2)
        m_and_Omega = tf.transpose(tf.stack([m, Omega]))
        # Compute Nakagami likelihood
        return Nakagami(tf.gather_nd(self.proximity_data, I), m_and_Omega)

#class BayesianIsoGPLVM(SVGP, InternalDataTrainingLossMixin):
#    def __init__(
#        self,
#        data: OutputData, #Should this be distance-based data OR OutputData
#        X_data_mean: tf.Tensor,
#        X_data_var: tf.Tensor,
#        kernel: Kernel,
#        num_inducing_variables: Optional[int] = None,
#        inducing_variable=None,
#        X_prior_mean=None,
#        X_prior_var=None,
#    ):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.
        :param data: data matrix, size N (number of points) x D (dimensions)
        :param X_data_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
            random permutation of X_data_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_data_mean.
        :param X_prior_var: prior variance used in KL term of bound. By default 1.
        """
#        num_data, num_latent_gps = X_data_mean.shape
#        super().__init__(kernel, likelihoods.Gaussian(), num_latent_gps=num_latent_gps)
#        self.data = data_input_to_tensor(data)
#        assert X_data_var.ndim == 2

#        self.X_data_mean = Parameter(X_data_mean)
#        self.X_data_var = Parameter(X_data_var, transform=positive())

#        self.num_data = num_data
#        self.output_dim = self.data.shape[-1]

#        assert np.all(X_data_mean.shape == X_data_var.shape)
#        assert X_data_mean.shape[0] == self.data.shape[0], "X mean and Y must be same size."
#        assert X_data_var.shape[0] == self.data.shape[0], "X var and Y must be same size."

#        if (inducing_variable is None) == (num_inducing_variables is None):
#            raise ValueError(
#                "BayesianGPLVM needs exactly one of `inducing_variable` and `num_inducing_variables`"
#            )

#        if inducing_variable is None:
            # By default we initialize by subset of initial latent points
            # Note that tf.random.shuffle returns a copy, it does not shuffle in-place
#            Z = tf.random.shuffle(X_data_mean)[:num_inducing_variables]
#            inducing_variable = InducingPoints(Z)

#        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

#        assert X_data_mean.shape[1] == self.num_latent_gps

        # deal with parameters for the prior mean variance of X
#        if X_prior_mean is None:
#            X_prior_mean = tf.zeros((self.num_data, self.num_latent_gps), dtype=default_float())
#        if X_prior_var is None:
#            X_prior_var = tf.ones((self.num_data, self.num_latent_gps))

#        self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float())
#        self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float())

#        assert self.X_prior_mean.shape[0] == self.num_data
#        assert self.X_prior_mean.shape[1] == self.num_latent_gps
#        assert self.X_prior_var.shape[0] == self.num_data
#        assert self.X_prior_var.shape[1] == self.num_latent_gps
 