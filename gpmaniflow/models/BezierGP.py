import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from gpflow.base import Parameter
from gpflow.models import BayesianModel
from gpflow.config import default_float
from gpflow.models.model import InputData, RegressionData
from gpflow.likelihoods import Likelihood
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin

from gpmaniflow.surfaces import BernsteinPolynomial, ControlPoints, AmortisedControlPoints
from gpmaniflow.utils import GetAllListPairs, GetNearGridPoints, GetNearGridPoints2, GetNearGridPoints3

class BezierProcess(BayesianModel, ExternalDataTrainingLossMixin): # Where should we inherit from?
    '''
    Implementation of a Bezier Stochastic Process. 
    '''
    def __init__(
            self,
            input_dim,
            likelihood: Likelihood, # Testing phase.
            order = 1,
            output_dim =1,
            amortise = False,
            num_data = None
            ):
        super().__init__()
        self.input_dim = input_dim
        self.order = order
        self.output_dim = output_dim
        self.num_data = num_data # Important to set when using minibatches
        self.likelihood = likelihood
        
        self.B = BernsteinPolynomial(self.order)
        if not amortise:
            self.P = ControlPoints(input_dim = self.input_dim, output_dim = self.output_dim, orders = self.order)
        else:
            self.P = AmortisedControlPoints(input_dim = self.input_dim, orders = self.order)
        self.amortise = amortise

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        X, Y = data
        
        if self.amortise:
            #I0 = self.order*tf.cast( GetAllListPairs(2, self.input_dim), default_float())
            #n00 = 2 ** self.input_dim
            #pd = tf.cast( tf.math.floor(self.P.batch_size ** (1/self.input_dim)), default_float())
                
            #I2 = GetAllListPairs(pd, self.input_dim)
            #I2 = (self.order) / pd * tf.cast(I2, default_float()) + tf.random.uniform(maxval = self.order, shape = [1,self.input_dim], dtype = default_float())
            #I2 = tf.math.floormod(I2, self.order - 1)
            #I2 = tf.floor(I2) + 1
            
            #global_I = tf.concat([I2, I0], 0)
            #print(global_I)

            pd = tf.cast( tf.math.floor(self.P.batch_size ** (1/self.input_dim)), default_float())
            #print(pd)
            if pd > self.order:
                I2 = tf.cast(GetAllListPairs(self.order+1, self.input_dim), default_float())
            else:
                I2 = GetAllListPairs(pd, self.input_dim)
                I2 = (self.order) / (pd - 1) * tf.cast(I2, default_float()) + tf.random.uniform(maxval = self.order + 1, shape = [1,self.input_dim], dtype = default_float())
                I2 = tf.math.floormod(I2, self.order + 1)
                I2 = tf.floor(I2)
            #I = tf.random.uniform(minval = 0, maxval = self.order + 1, shape = [self.P.batch_size, self.input_dim])
            global_I = I2
            
            #kl = self.P.kl_divergence(global_I)
            #print(kl)
            kl = self.P.num_points * tf.reduce_mean(self.P.kl_divergence(global_I))
            #print('KL:', kl)
            #kl = 0
            f_mean, f_var = self.predict_f(X, approx = True)
        else:
            kl = tf.reduce_sum(self.P.kl_divergence())
            f_mean, f_var = self.predict_f(X)
        
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        #var_exp = self.likelihood.predict_log_density(f_mean, f_var, Y) 
        if self.num_data is None:
            scale = 1
        else:
            scale = tf.cast(self.num_data, var_exp.dtype) / tf.cast(tf.shape(X)[0], var_exp.dtype)
        
        return scale * tf.reduce_sum(var_exp) - kl 

    def predict_f(self, Xnew: InputData, I = None, pi = None, approx = False, outB = None):       
        assert Xnew.shape[1] == self.input_dim
        
        if not self.amortise:
            I = GetAllListPairs(self.order+1,self.input_dim) 
            outB = self.B(Xnew)
            outB = tf.gather(outB, I, axis = 2) # [N, d, (order + 1)^d, d]
            outB = tf.linalg.diag_part(tf.transpose(outB, perm = (0,2,1,3) )) # [N, (order + 1)^d, d]
            outB = tf.reduce_prod(outB, axis = 2) # [N, (order + 1)^d]
            P_mean = tf.gather_nd(self.P.variational_posteriors.loc, I)
            P_var = tf.gather_nd(self.P.variational_posteriors.scale, I)
            f_mean = tf.matmul(outB, P_mean)
            f_var = tf.matmul(outB ** 2, P_var ** 2)
        
        if self.amortise:
            if approx is False:
                # PROBABLY THROW WARNING IF MANY CONTROL POINTS
                I = tf.data.Dataset.from_tensor_slices(
                        (GetAllListPairs(self.order+1, self.input_dim)))
                OutB = self.B(Xnew)
                f_mean = tf.zeros(shape = [tf.shape(Xnew)[0], self.output_dim], dtype = default_float())
                f_var = tf.zeros(shape = [tf.shape(Xnew)[0], self.output_dim], dtype = default_float())
                iterator = iter(I.batch(self.P.batch_size))
                for batch in iterator:
                    outB = tf.gather(OutB, batch, axis = 2)
                    outB = tf.linalg.diag_part(tf.transpose(outB, perm = (0,2,1,3) ))
                    outB = tf.reduce_prod(outB, axis = 2) # [N, B]
                    
                    amP = self.P.nn(batch / self.order)
                    #amP = self.P.nn(batch)
                    P_mean = amP[:,:,0]
                    P_var = tf.math.exp(amP[:,:,1])# * self.P.prior_sc(batch) # Std dev
                    
                    batch_f_mean = tf.matmul(outB, P_mean)
                    batch_f_var = tf.matmul(outB ** 2, P_var ** 2)

                    f_mean += batch_f_mean
                    f_var += batch_f_var

            else:
                if I is None:
                    #I = tf.random.uniform(minval = 0, maxval = self.order + 1, shape = [self.P.batch_size, self.input_dim])
                    #global_I = tf.math.floor(I)
                    ### 
                    # ALWAYS INCLUDE ENDPOINTS
                    #I0 = self.order*tf.cast( GetAllListPairs(2, self.input_dim), default_float())
                    #n00 = 2 ** self.input_dim
                    pd = tf.cast( tf.math.floor(self.P.batch_size ** (1/self.input_dim)), default_float())
                    if pd > self.order:
                        I2 = tf.cast(GetAllListPairs(self.order+1, self.input_dim), default_float())
                    else:
                        I2 = GetAllListPairs(pd, self.input_dim)
                        I2 = (self.order) / (pd - 1) * tf.cast(I2, default_float()) + tf.random.uniform(maxval = self.order + 1, shape = [1,self.input_dim], dtype = default_float())
                        I2 = tf.math.floormod(I2, self.order + 1)
                        I2 = tf.floor(I2)
                    global_I = I2
                

                if outB is None:
                    outB = self.B(Xnew) # [N, d, (order + 1)]
                
                #sumBfvTotal = tf.reduce_sum(tf.reduce_prod(outB ** 2, axis = 1), axis = 1)
                sumBfvTotal = tf.reduce_prod(tf.reduce_sum(outB ** 2, axis = 2), axis = 1)
                n = 2000
                #local_I = GetNearGridPoints2(Xnew, order = self.order, d = self.input_dim, n = n) # [N, n, d]
                local_I = GetNearGridPoints3(outB, self.input_dim, self.order, n = n)
                # CAN THIS FUNCTION NOT HAVE KNOWN SHAPE WHEN CONSTRUCTING THE GRAPH
                #print(local_I.shape.ndims)
                #print(local_I.shape)
                local_outB = tf.gather(outB, tf.cast(local_I, dtype = tf.int32), 
                        axis = 2, batch_dims = 1)
                #print(local_outB.shape)
                local_outB = local_outB.to_tensor(default_value = -1)
                #print(local_outB.shape)
                local_outB = tf.linalg.diag_part(tf.transpose(local_outB, perm = (0,2,1,3) ))
                #print(local_outB.shape)
                local_outB = tf.RaggedTensor.from_tensor(local_outB, padding = [-1] * self.input_dim, ragged_rank = 1)
                #local_outB = tf.gather(local_outB, range(self.input_dim),
                #    batch_dims = 1)
                #print(local_outB.shape)
                local_outB = tf.reduce_prod(local_outB, axis = 2) # [N, n]
                local_outB = tf.expand_dims(local_outB, axis = 1) # Only for matmul
                
                sumBfm = tf.reduce_sum(local_outB, axis = 2) # [N, 1]
                sumBfv = tf.reduce_sum(local_outB ** 2, axis = 2) # [N, 1]
                local_amP = tf.map_fn(lambda x: self.P.nn(x), local_I / self.order, fn_output_signature = tf.RaggedTensorSpec(shape = [None, 1, 2], ragged_rank = 0, dtype = default_float())) # [N, n, out_dim, 2] # IS THERE A MORE EFFICIENT WAY HERE
                #local_amP = tf.map_fn(lambda x: self.P.nn(x), local_I / self.order) # [N, n, out_dim, 2] # IS THERE A MORE EFFICIENT WAY HERE

                local_P_mean = local_amP[:,:,:,0] # [N, n, out_dim]
                #new = self.P.prior_sc(tf.cast(local_I, tf.int32))
                #new = tf.map_fn(self.P.prior_sc, tf.cast(local_I, tf.int32), fn_output_signature = default_float())
                #print('new:', new)
                #print(local_I[0,:,:])
                local_P_var = tf.math.exp(local_amP[:,:,:,1])# * new # [N, n, out_dim] # Std dev
                #print(local_P_mean.shape)
                #print(local_P_var.shape)
                local_f_mean = tf.squeeze(tf.matmul(local_outB, local_P_mean), axis = 1) # [N, out_dim]
                local_f_var = tf.squeeze(tf.matmul(local_outB ** 2, local_P_var ** 2), axis = 1) 

                global_outB = tf.gather(outB, tf.cast(global_I, dtype = tf.int32), axis = 2)
                global_outB = tf.linalg.diag_part(tf.transpose(global_outB, perm = (0,2,1,3) ))
                global_outB = tf.reduce_prod(global_outB, axis = 2) # [N, B]
                #print(global_outB.shape) 
                sumBfvTotal = tf.expand_dims(sumBfvTotal, axis = 1)
                
                global_sumBfm = tf.reduce_sum(global_outB, axis = 1, keepdims = True) # [N, 1]
                global_sumBfv = tf.reduce_sum(global_outB ** 2, axis = 1, keepdims = True) # [N, 1]
                amP = self.P.nn(global_I / self.order) # nn forward pass
                #amP = self.P.nn(global_I / self.order) # nn forward pass
                P_mean = amP[:,:,0] # [B, out_dim]
                #new2 = self.P.prior_sc(tf.cast(global_I, tf.int32))
                #print('new2:', new2)
                P_var = tf.math.exp(amP[:,:,1])# * new2 # [B, out_dim] # Std dev

                
                global_f_mean = tf.matmul(global_outB, P_mean) # [N, out_dim]
                global_f_var = tf.matmul(global_outB ** 2, P_var ** 2) # Here we dont square P_var
                #print(local_f_mean.shape)
                #print(global_sumBfm.shape)
                #print((1. - sumBfm)/global_sumBfm * global_f_mean) 
                #print((1 - sumBfm) * (self.P.num_points - n) / global_outB.shape[1] * global_f_mean)
                print('sumBfm:', sumBfm)
                #print(tf.reduce_mean(1/global_sumBfm))
                #print(sumBfvTotal - sumBfv)
                #print(global_sumBfv)
                f_mean = local_f_mean + (1. - sumBfm) / global_sumBfm  * global_f_mean
                #f_mean = local_f_mean + (1 - sumBfm) * (self.P.num_points - n)/global_outB.shape[1] * global_f_mean
                f_var = local_f_var + (sumBfvTotal - sumBfv) / global_sumBfv * global_f_var
                f_mean = f_mean.to_tensor()
                f_var = f_var.to_tensor()
                #print(f_mean.shape)
                #print(f_var.shape)
        return f_mean, f_var
    
    def low_variance_predict_f(self, Xnew: InputData):
        # Xnew should only be 1 point (?)
        local_I = GetNearGridPoints(tf.expand_dims(Xnew, axis = 0), self.order, n = 100)  
        #print(local_I)
        outB = self.B(tf.expand_dims(Xnew, axis=0))

        sumBfvTotal = tf.reduce_sum(outB ** 2, axis = 2) # [1,d] # Should maybe return [1]?
        sumBfvTotal = tf.reduce_prod(sumBfvTotal)
        local_amP = self.P.nn(local_I / self.order)
        local_outB = tf.gather(outB, tf.cast(local_I, dtype = tf.int32), axis = 2)
        local_outB = tf.linalg.diag_part(tf.transpose(local_outB, perm = (0,2,1,3) ))
        local_outB = tf.reduce_prod(local_outB, axis = 2) # [N, B]
        
        sumBfm = tf.reduce_sum(local_outB)
        sumBfv = tf.reduce_sum(local_outB ** 2)

        local_f_mean = tf.matmul(local_outB, local_amP[:,:,0])
        local_f_var = tf.matmul(local_outB ** 2, tf.exp(local_amP[:,:,1])) 
        return local_f_mean, local_f_var, sumBfm, sumBfv, sumBfvTotal

if __name__ == '__main__':
    I = GetAllListPairs(6, 3)
    #print(I)
    from gpflow.likelihoods import Gaussian
    m = BezierProcess(input_dim=3, order = 5, output_dim = 1, likelihood = Gaussian(), amortise = True)
    from gpflow.utilities import print_summary
    print_summary(m)
    import numpy as np
    Xnew = np.array([[1., 1., 1.], [0.5, 0.5, 0.5], [0.0, 0., 0.]])
    #print(Xnew)
    f_mean, f_var = m.predict_f(Xnew)
    kl = m.P.kl_divergence(I)
    #print(kl)
    #print(f_mean)
    #print(f_var)
    I = GetAllListPairs(10, 3)
    print(I) 
    I = 31 / 10 * tf.cast(I, tf.float64) + tf.random.uniform(maxval = 31, shape = [1], dtype = tf.float64)
    print(I)
    I = tf.math.floormod(I, 31)
    I = tf.math.floor(I)
    print(I)
