from gpflow.conditionals.dispatch import conditional
from gpflow.config import default_jitter
from gpflow.conditionals.util import base_conditional, expand_independent_outputs
from gpflow.inducing_variables import InducingVariables
from .. import covariances
from ..kernels import dSquaredExponential

@conditional.register(object, InducingVariables, dSquaredExponential, object)
def _conditional(Xnew, inducing_variable, kernel, f, *, full_cov=False, full_output_cov=False, q_sqrt=None,
                               white=False):
    Kmm = covariances.Kuu(inducing_variable, kernel, jitter=default_jitter())
    Kmn = covariances.Kuf(inducing_variable, kernel, Xnew)
    Knn = kernel(Xnew, full_cov=full_cov)
    
    fmean, fvar = base_conditional(
        Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )
    return fmean, expand_independent_outputs(fvar, full_cov, full_output_cov)