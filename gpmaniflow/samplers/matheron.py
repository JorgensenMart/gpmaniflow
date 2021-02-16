import tensorflow as tf
import numpy as np
from gpflow.config import default_float
from gpflow.utilities import Dispatcher
from gpflow.kernels import SquaredExponential
from gpflow import covariances, default_jitter
from gpmaniflow.kernels import dSquaredExponential
from gpmaniflow.utils import Kronecker

sample_matheron = Dispatcher("sample_matheron")

@sample_matheron.register(object, SquaredExponential, object)
def _sample_matheron(inducing_variable, kernel, q_mu, q_sqrt = None, likelihood_var = None, white = True, num_samples = 1, num_basis = 512):
    if not isinstance(kernel, SquaredExponential):
        raise NotImplementedError

    # Draw from prior
    bias = tf.random.uniform(shape = [num_basis], maxval=2*np.pi, dtype = q_mu.dtype)
    spectrum = tf.random.normal(shape = [num_basis, inducing_variable.Z.shape[1]], dtype = q_mu.dtype)
    
    _Z = tf.divide(inducing_variable.Z, kernel.lengthscales)
    w = tf.random.normal(shape = [num_samples, q_mu.shape[1], num_basis, 1], dtype = q_mu.dtype) # [S, P, num_basis, 1]
    phiZ = tf.sqrt(kernel.variance * 2 / num_basis) * tf.cos(tf.matmul(_Z, spectrum, transpose_b = True) + bias) # [M, num_basis]
    fZ = tf.transpose(tf.tensordot(phiZ, w, [[1], [2]]), perm = [1,2,0,3])
    #fZ = fZ + tf.sqrt(tf.cast(default_jitter(), tf.float64)) * tf.random.normal(shape=fZ.shape, dtype=fZ.dtype) # for stability
    
    # Update
    #eps = tf.random.normal(shape = [num_samples, q_mu.shape[1], q_mu.shape[0], 1], dtype = q_mu.dtype) # [S, P, M, 1]
    #_u = tf.tile(tf.linalg.adjoint(q_mu)[None, :, :, None], [num_samples, 1,1,1]) +  tf.matmul(tf.tile(q_sqrt[None,:,:,:], [num_samples,1,1,1]), eps)# [S, P, M, 1]
    
    #Luu = tf.linalg.cholesky(covariances.Kuu(inducing_variable, kernel, jitter=default_jitter())) # [M, M]
    #Luu = tf.tile(Luu[None, None, :, :], [num_samples, q_mu.shape[1],1,1]) # [S, P, M, M]
    if q_sqrt is not None:
        eps = tf.random.normal(shape = [num_samples, q_mu.shape[1], q_mu.shape[0], 1], dtype = q_mu.dtype) # [S, P, M, 1]
        Luu = tf.linalg.cholesky(covariances.Kuu(inducing_variable, kernel, jitter=default_jitter())) # [M, M]
        Luu = tf.tile(Luu[None, None, :, :], [num_samples, q_mu.shape[1],1,1]) # [S, P, M, M]
        _u = tf.tile(tf.linalg.adjoint(q_mu)[None, :, :, None], [num_samples, 1,1,1]) +  tf.matmul(tf.tile(q_sqrt[None,:,:,:], [num_samples,1,1,1]), eps)# [S, P, M, 1]
        
        if white:
            _u = Luu @ _u 
    else:
        Luu = tf.linalg.cholesky(covariances.Kuu(inducing_variable, kernel, jitter=likelihood_var))
        Luu = tf.tile(Luu[None, None, :, :], [num_samples, q_mu.shape[1],1,1]) # [S, P, M, M]
        _u = tf.tile(tf.linalg.adjoint(q_mu)[None, :, :, None], [num_samples, 1,1,1])# +  tf.matmul(Luu, eps)# [S, P, M, 1]
        fZ = fZ + tf.sqrt(likelihood_var) * tf.random.normal(shape = fZ.shape, dtype = fZ.dtype)


    res_upd = tf.linalg.cholesky_solve(Luu, _u - fZ) # [S, P, M, 1]
    def sampler(Xnew):
        phiX = tf.sqrt(kernel.variance * 2 / num_basis) * tf.cos(tf.matmul(tf.divide(Xnew,kernel.lengthscales), spectrum, transpose_b = True) + bias) # [N, num_basis]
        fX = tf.transpose(tf.tensordot(phiX, w, [[1], [2]]), perm = [1,2,0,3])
        f_upd = kernel.K(Xnew, inducing_variable.Z) @ res_upd # [S, P, N, 1]
        samples = tf.linalg.adjoint( tf.squeeze(fX + f_upd, axis = 3) )
        return samples # [S, N, P]
    return sampler

@sample_matheron.register(object, dSquaredExponential, object)
def _sample_matheron(inducing_variable, kernel, q_mu, q_sqrt = None, likelihood_var = None, white = True, num_samples = 1, num_basis = 512):

    # Draw from prior
    bias = tf.random.uniform(shape = [num_basis], maxval=2*np.pi, dtype = q_mu.dtype)
    spectrum = tf.random.normal(shape = [num_basis, inducing_variable.Z.shape[1]], dtype = q_mu.dtype)
    
    _Z = tf.divide(inducing_variable.Z, kernel.lengthscales)
    w = tf.random.normal(shape = [num_samples, q_mu.shape[1], num_basis, 1], dtype = q_mu.dtype) # [S, P, num_basis, 1]
    phiZ = tf.sqrt(kernel.variance * 2 / num_basis) * tf.cos(tf.matmul(_Z, spectrum, transpose_b = True) + bias) # [M, num_basis]
    fZ = tf.transpose(tf.tensordot(phiZ, w, [[1], [2]]), perm = [1,2,0,3])
    #fZ = fZ + tf.sqrt(tf.cast(default_jitter(), tf.float64)) * tf.random.normal(shape=fZ.shape, dtype=fZ.dtype) # for stability
    
    # Update
    if q_sqrt is not None:
        eps = tf.random.normal(shape = [num_samples, q_mu.shape[1], q_mu.shape[0], 1], dtype = q_mu.dtype) # [S, P, M, 1]
        Luu = tf.linalg.cholesky(covariances.Kuu(inducing_variable, kernel, jitter=default_jitter())) # [M, M]
        Luu = tf.tile(Luu[None, None, :, :], [num_samples, q_mu.shape[1],1,1]) # [S, P, M, M]
        _u = tf.tile(tf.linalg.adjoint(q_mu)[None, :, :, None], [num_samples, 1,1,1]) +  tf.matmul(tf.tile(q_sqrt[None,:,:,:], [num_samples,1,1,1]), eps)# [S, P, M, 1]
        
        if white:
            _u = Luu @ _u 
    else:
        Luu = tf.linalg.cholesky(covariances.Kuu(inducing_variable, kernel, jitter=likelihood_var))
        Luu = tf.tile(Luu[None, None, :, :], [num_samples, q_mu.shape[1],1,1]) # [S, P, M, M]
        _u = tf.tile(tf.linalg.adjoint(q_mu)[None, :, :, None], [num_samples, 1,1,1])# +  tf.matmul(Luu, eps)# [S, P, M, 1]
        fZ = fZ + tf.sqrt(likelihood_var) * tf.random.normal(shape = fZ.shape, dtype = fZ.dtype)

    res_upd = tf.linalg.cholesky_solve(Luu, _u - fZ) # [S, P, M, 1]
    def sampler(Xnew):
        phiX = - tf.sqrt(kernel.variance * 2 / num_basis) * tf.divide(tf.transpose(spectrum), kernel.lengthscales) #[d, num_basis]
        phiX = Kronecker(tf.ones([Xnew.shape[0], 1], dtype = default_float()), phiX) #[Nd, num_basis]
        phiX = phiX * Kronecker(tf.sin(tf.matmul(tf.divide(Xnew,kernel.lengthscales), spectrum, transpose_b = True) + bias),tf.ones([Xnew.shape[1], 1], dtype = default_float()))

        fX = tf.transpose(tf.tensordot(phiX, w, [[1], [2]]), perm = [1,2,0,3])
        fX = tf.reshape(fX, [num_samples, -1, Xnew.shape[0], Xnew.shape[1]]) # [S, P, N, d]

        f_upd = kernel.K(Xnew, inducing_variable) @ res_upd # [S, P, N, 1] #[S, P, Nd, 1]
        f_upd = tf.reshape(f_upd, [num_samples, -1, Xnew.shape[0], Xnew.shape[1]]) #[S, P, N, d]

        samples = tf.transpose(fX + f_upd, perm = [0, 2, 3, 1])
        return samples # [S, N, d, P]
    return sampler