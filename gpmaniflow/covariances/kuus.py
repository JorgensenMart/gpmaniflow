import tensorflow as tf
from gpflow.inducing_variables import InducingPoints
from gpmaniflow.kernels import dSquaredExponential
from gpflow.covariances.dispatch import Kuu

@Kuu.register(InducingPoints, dSquaredExponential)
def Kuu_kernel_inducingpoints(inducing_variable: InducingPoints, kernel: dSquaredExponential, *, jitter=0.0):
    Kzz = kernel.K(inducing_variable)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz