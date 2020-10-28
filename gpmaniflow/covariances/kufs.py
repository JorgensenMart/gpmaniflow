from gpflow.base import TensorLike
from gpflow.inducing_variables import InducingPoints
from gpmaniflow.kernels import dSquaredExponential
from gpflow.covariances.dispatch import Kuf

@Kuf.register(InducingPoints, dSquaredExponential, TensorLike)
def Kuf_kernel_inducingpoints(inducing_variable: InducingPoints, kernel: dSquaredExponential, Xnew, jitter=0.0):
    Kzx = kernel.K(inducing_variable, Xnew)
    return Kzx