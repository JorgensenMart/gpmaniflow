import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from gpflow.config import default_float
from gpflow.models.model import InputData
from gpflow.base import Parameter, Module
from gpflow.utilities import positive

from gpmaniflow.utils import binomial_coef as binom
from gpmaniflow.utils import GetAllListPairs

class BernsteinPolynomial():
    def __init__(self, orders = None):
        self.orders = orders
    
    def __call__(self, Xnew: InputData):
        return self.forward(Xnew)

    def forward(self, Xnew: InputData):
        O = self.orders + 1
        Xnew = tf.stack([Xnew] * O, axis = 2)
        X1 = tf.math.pow(Xnew, range(O))
        X2 = tf.math.pow(1. - Xnew, np.array(self.orders) - range(O))
        B_out = X1 * X2 * binom(np.array(self.orders), range(O))
        return B_out


class LogBezierButtress(tf.keras.layers.Layer):
    def __init__(self, input_dim = 1, orders = 1, muN = 0.8, sigma2N = 3, perm = None, num_perm = 20):
        super(LogBezierButtress,self).__init__()
        self.orders = orders
        _dummy = [0]; _dummy.extend(orders)
        self.input_dim = input_dim
        self.num_perm = num_perm
        
        self.muN = muN
        self.sigma2N = sigma2N
        
        meanw_init = tf.random_normal_initializer(mean = (tf.math.log(tf.cast(self.muN, default_float())/ self.num_perm)) ** (1/self.input_dim), stddev = 0.1)
        #meanw_init = tf.zeros_initializer()
        
        varw_init = tf.random_normal_initializer(mean = tf.math.log(tf.cast(self.sigma2N, default_float())), stddev = 0.05)
        #varw_init = tf.zeros_initializer()
        if perm is None:
            _perm = np.tile(range(self.input_dim), (self.num_perm,1))
            self.perm = np.apply_along_axis(np.random.permutation, axis=1, arr=_perm)
        else:
            self.perm = perm
        self.meanw = (*[tf.Variable(initial_value = meanw_init(shape = (self.num_perm, _dummy[i] + 1, _dummy[i+1] + 1), dtype = default_float()), trainable = True,) for i in range(input_dim)],)
        self.varw = (*[tf.Variable(initial_value = varw_init(shape = (self.num_perm, _dummy[i] + 1, _dummy[i+1] + 1), dtype = default_float()), trainable = True,) for i in range(input_dim)],)
        
        self.B = BernsteinPolynomial(self.orders[0])
        #self.prior_sc = tf.ones(self.orders[0] + 1, default_float())#prior_adjusting(self.orders[0])
        self.prior_sc = prior_adjusting(self.orders[0])
        #self.prior_variance = Parameter(value = [1 / (self.num_perm)] * self.num_perm, dtype = default_float(), trainable = True, transform = positive())
        #self.posterior_precision = Parameter(value = [self.num_perm] * self.num_perm, dtype = default_float(), trainable = True, transform = positive()) 
    
    def f_mean(self, Xnew):
        outB = self.B(Xnew) # [N, d, o + 1]
        outB = tf.gather(outB, self.perm, axis = 1)
        outB = tf.transpose(outB, (1,0,2,3)) # [R, N, d, o + 1]
        f = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1] 
        for i in range(self.input_dim):
            f = tf.matmul(f, tf.math.exp(self.meanw[i])) # [R, N, o+1]
            f = f * outB[:,:,i,:] # [R, N, o + 1]
        f = tf.reduce_sum(f, axis = -1, keepdims = True)
        f = tf.reduce_sum(f, axis = 0)
        return f# [N, 1]

    def f_var(self, Xnew):
        outB = self.B(Xnew) # [N, d, o + 1]
        outB = tf.gather(outB, self.perm, axis = 1)
        outB = tf.transpose(outB, (1,0,2,3)) # [R, N, d, o + 1]
        #f1 = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1]
        #f2 = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1]
        meanterm = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1]
        one = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1]
        two = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1]
        three = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1]
        four = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1]
        five = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1]
        six = tf.ones((self.num_perm, tf.shape(outB)[1], 1), dtype = default_float()) # [R, N, 1]
        
        for i in range(self.input_dim):
            #f = tf.matmul(f, tf.math.exp(self.meanw[i]) ** 2  * tf.math.exp(self.varw[i]) * tf.transpose(self.prior_sc ** 2)) # [R, N, o+1]
            #f1 = tf.matmul(f1, tf.math.exp(self.meanw[i]) ** 2  * (1.0 + tf.math.exp(self.varw[i])) ) # [R, N, o+1]
            one = tf.matmul(one, tf.math.exp(self.meanw[i]) ** 2  * tf.math.exp(self.varw[i])) # [R, N, o+1]
            two = tf.matmul(two, tf.math.exp(self.meanw[i]) ** 2  * tf.math.exp(self.varw[i]) ** 2 ) # [R, N, o+1]
            three = tf.matmul(three, tf.math.exp(self.meanw[i]) ** 2  * tf.math.exp(self.varw[i]) ** 3 ) # [R, N, o+1]
            four = tf.matmul(four, tf.math.exp(self.meanw[i]) ** 2  * tf.math.exp(self.varw[i]) ** 4 ) # [R, N, o+1]
            five = tf.matmul(five, tf.math.exp(self.meanw[i]) ** 2  * tf.math.exp(self.varw[i]) ** 5 ) # [R, N, o+1]
            six = tf.matmul(six, tf.math.exp(self.meanw[i]) ** 2  * tf.math.exp(self.varw[i]) ** 6 ) # [R, N, o+1]
            #f2 = tf.matmul(f2, tf.math.exp(self.meanw[i]) ** 2 ) # [R, N, o+1]
            meanterm = tf.matmul(meanterm, tf.math.exp(self.meanw[i]) ** 2 ) # [R, N, o+1]
            
            one = one * outB[:,:,i,:] ** 2 # [R, N, o + 1]
            two = two * outB[:,:,i,:] ** 2 # [R, N, o + 1]
            three = three * outB[:,:,i,:] ** 2 # [R, N, o + 1]
            four = four * outB[:,:,i,:] ** 2 # [R, N, o + 1]
            five = five * outB[:,:,i,:] ** 2 # [R, N, o + 1]
            six = six * outB[:,:,i,:] ** 2 # [R, N, o + 1]
            #f1 = f1 * outB[:,:,i,:] ** 2 # [R, N, o + 1]
            #f2 = f2 * outB[:,:,i,:] ** 2 # [R, N, o + 1]
            meanterm = meanterm * outB[:,:,i,:] ** 2 # [R, N, o + 1]
        
        one = tf.reduce_sum(one, axis = -1, keepdims = True) # [R, N, 1]
        two = tf.reduce_sum(two, axis = -1, keepdims = True) # [R, N, 1]
        three = tf.reduce_sum(three, axis = -1, keepdims = True) # [R, N, 1]
        four = tf.reduce_sum(four, axis = -1, keepdims = True) # [R, N, 1]
        five = tf.reduce_sum(five, axis = -1, keepdims = True) # [R, N, 1]
        six = tf.reduce_sum(six, axis = -1, keepdims = True) # [R, N, 1]
        
        meanterm = tf.reduce_sum(meanterm, axis = -1, keepdims = True) # [R, N, 1]
        #f1 = tf.reduce_sum(f1, axis = -1, keepdims = True) # [R, N, 1]
        #f2 = tf.reduce_sum(f2, axis = -1, keepdims = True) # [R, N, 1]
        #f = tf.reduce_sum(f / self.posterior_precision[:,None,None], axis = 0)
        #f = tf.reduce_sum(f1, axis = 0) - tf.reduce_sum(f2, axis = 0)
        f = tf.reduce_sum(one + 1/2 * two + 1/6 * three + 1/24 * four + 1/120 * five + 1/720 * six, axis = 0)# - tf.reduce_sum(meanterm, axis = 0)
        #f = tf.reduce_sum(one + 1/2 * two + 1/6 * three + 1/24, axis = 0)# - tf.reduce_sum(meanterm, axis = 0)
        
        return f # [N, 1]

    def kl(self):
        muN = self.muN / self.num_perm
        sigma2N = self.sigma2N / self.num_perm ** 2
        #beta = np.log(1 + sigma2N/muN**2)
        a = 100.0
        
        num_points = tf.reduce_prod(tf.cast(self.orders, dtype = default_float()) + 1.0)
        beta = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        invbeta = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        I211 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        I212 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        I221 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        I222 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        I231 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        I232 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        I233 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  

        meanterm2 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        traceterm = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        detterm = 0.0
        detterm2 = 0.0
        for i in range(self.input_dim):
            traceterm = tf.matmul(traceterm, tf.math.exp(self.varw[i]) * tf.transpose(1 / self.prior_sc **2)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            detterm += tf.reduce_mean(self.varw[i])
            detterm2 += tf.reduce_mean(tf.math.log(tf.ones_like(self.varw[i]) * tf.transpose(self.prior_sc ** 2)))
            beta = tf.matmul(beta, tf.ones_like(self.varw[i]) * tf.transpose(self.prior_sc **2)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            invbeta = tf.matmul(invbeta, tf.ones_like(self.varw[i]) * tf.transpose(1 / self.prior_sc **2)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            I211 = tf.matmul(I211, (muN ** (1/self.input_dim)/tf.math.exp(self.meanw[i])) ** (2.0/a) * tf.transpose(1 / self.prior_sc **2)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            I212 = tf.matmul(I212, (muN ** (1/self.input_dim)/tf.math.exp(self.meanw[i])) ** (1.0/a) * tf.transpose(1 / self.prior_sc **2)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            I221 = tf.matmul(I221, tf.math.exp(self.varw[i]) ** 2.0 * tf.transpose(1 / self.prior_sc **2)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            I222 = tf.matmul(I222, tf.math.exp(self.varw[i])) / (tf.cast(self.orders[i], default_float()) + 1.0)
            I231 = tf.matmul(I231, tf.math.exp(self.varw[i]) * (muN ** (1/self.input_dim) / tf.math.exp(self.meanw[i])) ** (1.0/a) * tf.transpose(1 / self.prior_sc ** 2)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            I232 = tf.matmul(I232, tf.math.exp(self.varw[i]) *tf.transpose( 1/ self.prior_sc ** 2)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            I233 = tf.matmul(I233, (muN ** (1/self.input_dim)/tf.math.exp(self.meanw[i])) ** (1.0/a)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            #I233 = tf.matmul(I233, tf.math.exp(self.meanw[i]) ** (1.0/a)) / (tf.cast(self.orders[i], default_float()) + 1.0)
        
        traceterm = tf.reduce_sum(traceterm, axis = (1,2))
        traceterm = tf.reduce_mean(traceterm)
        beta = tf.reduce_sum(beta, axis = (1,2))
        beta = tf.reduce_mean(beta)
        invbeta = tf.reduce_sum(invbeta, axis = (1,2))
        invbeta = tf.reduce_mean(invbeta)

        I211 = tf.reduce_sum(I211, axis = (1,2))
        I211 = tf.reduce_mean(I211)
        I212 = tf.reduce_sum(I212, axis = (1,2))
        I212 = tf.reduce_mean(I212)
        I221 = tf.reduce_sum(I221, axis = (1,2))
        I221 = tf.reduce_mean(I221)
        I222 = tf.reduce_sum(I222, axis = (1,2))
        I222 = tf.reduce_mean(I222)
        I231 = tf.reduce_sum(I231, axis = (1,2))
        I231 = tf.reduce_mean(I231)
        I232 = tf.reduce_sum(I232, axis = (1,2))
        I232 = tf.reduce_mean(I232)
        I233 = tf.reduce_sum(I233, axis = (1,2))
        I233 = tf.reduce_mean(I233)
        
        I21 = a**2 * (I211 + invbeta - 2*I212)
        I22 = 0.25 * (I221 + beta - 2*I222)
        I23 = a*(I231 - I232) - a*(I233 - 1)
        
        meanterm = I21 + I22 + I23
        traceterm = traceterm
        detterm = detterm2 - detterm
        #print(beta)
        #print(meanterm)
        #print(traceterm)
        #print(detterm)
        return num_points*0.5*(traceterm - 1.0 + meanterm + detterm) 

    """
    def kl(self):
        a = 1000.0 # HIGHER SHOULD GIVE BETTER APPROXIMATE LOGARITHMS 
        beta = 18.4207 
        mu_p = 1.0

        num_points = tf.reduce_prod(tf.cast(self.orders, dtype = default_float()) + 1.0)
        meanterm1 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        meanterm2 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        traceterm = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        detterm0 = 0.0
        detterm1 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        detterm2 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        detterm3 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        detterm10 = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        
        #detterm_lb = tf.ones((self.num_perm, 1, 1), dtype = default_float())  
        detterm_lb = 0.0
        for i in range(self.input_dim):
            traceterm = tf.matmul(traceterm, (1.0 + tf.math.exp(self.varw[i]))**(1.0 / a)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            meanterm1 = tf.matmul(meanterm1, (1.0 + tf.math.exp(self.varw[i])) ** (1.0 / (2*a)) / tf.math.exp(self.meanw[i]) ** (1.0 / a) ) / (tf.cast(self.orders[i], default_float()) + 1.0)
            meanterm2 = tf.matmul(meanterm2, (1.0 + tf.math.exp(self.varw[i])) ** (2.0 / (2*a)) / tf.math.exp(self.meanw[i]) ** (2.0 / a) ) / (tf.cast(self.orders[i], default_float()) + 1.0)
            # SHOULD THERE BE SOME PRIOR ADJUSTMENT
            detterm0 += tf.reduce_mean((1.0/a) * tf.math.log(1.0 + tf.math.exp(self.varw[i])))# * self.num_perm 
            detterm1 = tf.matmul(detterm1, 1.0 / (1.0 + tf.math.exp(self.varw[i])) ** (1.0 / a)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            detterm2 = tf.matmul(detterm2, 1.0 / (1.0 + tf.math.exp(self.varw[i])) ** (2.0 / a)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            detterm3 = tf.matmul(detterm3, 1.0 / (1.0 + tf.math.exp(self.varw[i])) ** (3.0 / a)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            detterm10 = tf.matmul(detterm10, 1.0 / (1.0 + tf.math.exp(self.varw[i])) ** (10.0 / a)) / (tf.cast(self.orders[i], default_float()) + 1.0) 
            #detterm_lb = tf.matmul(detterm_lb, 1.0 / (1.0 + tf.math.exp(self.varw[i])) ** (1.0/a)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            #detterm_lb = tf.matmul(detterm_lb, 1.0 / (1.0 + tf.math.exp(self.varw[i])) ** (1.0/a)) / (tf.cast(self.orders[i], default_float()) + 1.0)
            detterm_lb += tf.reduce_mean(1.0 / tf.math.log((1.0 + tf.math.exp(self.varw[i]))))# / (tf.cast(self.orders[i], default_float()) + 1.0)
            
        traceterm = tf.reduce_sum(traceterm, axis = (1,2))
        traceterm = tf.reduce_mean(traceterm)
        meanterm1 = tf.reduce_sum(meanterm1, axis = (1,2))
        meanterm1 = tf.reduce_mean(meanterm1)####, axis = (1,2))
        meanterm2 = tf.reduce_sum(meanterm2, axis = (1,2))
        meanterm2 = tf.reduce_mean(meanterm2)#, axis = (1,2))
        detterm1 = tf.reduce_sum(detterm1, axis = (1,2))
        detterm1 = tf.reduce_mean(detterm1)#, axis = (1,2))
        detterm2 = tf.reduce_sum(detterm2, axis = (1,2))
        detterm2 = tf.reduce_mean(detterm2)#, axis = (1,2))
        detterm3 = tf.reduce_sum(detterm3, axis = (1,2))
        detterm3 = tf.reduce_mean(detterm3)#, axis = (1,2))
        detterm10 = tf.reduce_sum(detterm10, axis = (1,2))
        detterm10 = tf.reduce_mean(detterm10)#, axis = (1,2))
        #detterm_lb = tf.reduce_sum(detterm_lb, axis = (1,2))
        #detterm_lb = tf.reduce_mean(detterm_lb)#, axis = (1,2))
        
        #print(a)
        #print(beta)
        print('tr:', traceterm)
        traceterm = a * traceterm / beta - a / beta
        meanterm = a ** 2 / beta * ((mu_p / (tf.cast(tf.math.exp(beta), dtype = default_float()) ** 0.5)) ** (2.0/a) * meanterm2 + 1.0 - 2 * (mu_p/(tf.cast(tf.math.exp(beta), dtype = default_float())**0.5)) **(1.0/a) * meanterm1)
        print(detterm0)
        print(detterm1)
        print(detterm2)
        print(detterm3)
        print(detterm10)
        print('lb:', detterm_lb)
        detterm = tf.cast(tf.math.log(beta) - tf.math.log(a), dtype = default_float()) - (detterm0 - detterm1 - 0.5 * detterm2 - 1.0 / 3.0 * detterm3 - 1.0 / 10.0 * detterm10)
        print(detterm)
        print(tf.cast(tf.math.log(beta) - tf.math.log(a), dtype = default_float()))
        detterm = tf.cast(tf.math.log(beta), dtype = default_float()) - (1 - (detterm_lb))
        print('trace:', traceterm)
        #print(meanterm1)
        #print(meanterm2)
        print('mean:', meanterm)
        print('det:', detterm)
        #traceterm = 0.0
        #detterm = 0.0
        #meanterm = 0.0
        #return 0.5 * (traceterm + meanterm - 1.0 + tf.reduce_sum(tf.math.log(self.prior_variance * self.posterior_precision)) - detterm)
        return 0.5 * (traceterm + meanterm - 1.0 + detterm)
    """

    def P_mean(self, batch):
        P = tf.ones((self.num_perm, tf.shape(batch)[0],1), dtype = default_float()) # [R, N, 1]
        for i in range(self.input_dim):
            P = tf.matmul(P, self.meanw[i]) # [R, N, o + 1]
            K = tf.one_hot(batch[:,i], self.orders[i] + 1, dtype = default_float())
            K = tf.tile(K[None, :, :], [self.num_perm, 1, 1])
            P = P * K
        P = tf.reduce_sum(P, axis = -1, keepdims = True)
        P = tf.reduce_sum(P, axis = 0)
        return P# [N, 1}

    def P_var(self, batch):
        P = tf.ones((self.num_perm, tf.shape(batch)[0],1), dtype = default_float()) # [R, N, 1]
        for i in range(self.input_dim):
            P = tf.matmul(P, tf.math.exp(self.varw[i]) * tf.transpose(self.prior_sc ** 2)) 
            K = tf.one_hot(batch[:,i], self.orders[i] + 1, dtype = default_float())
            K = tf.tile(K[None, :, :], [self.num_perm, 1, 1])
            P = P * K
        P = tf.reduce_sum(P, axis = -1, keepdims = True)
        P = tf.reduce_sum(P, axis = 0)
        return P # [N, 1]

    def integral(self):
        num_points = tf.reduce_prod(tf.cast(self.orders, dtype = default_float()) + 1) 
        f = tf.ones((self.num_perm, 1, 1), dtype = default_float()) # [R, N, 1] 
        for i in range(self.input_dim):
            f = tf.matmul(f, tf.math.exp(self.meanw[i])) # [R, N, o+1]
        f = tf.reduce_sum(f)
        return f / num_points

def prior_adjusting(order):
    muN = 0.8 / 20#self.num_perm
    sigma2N = 3 / 20 #self.num_perm ** 2
    #if order > 25:
    #    raise NotImplementedError
    #I = tf.cast(GetAllListPairs(order + 1, 1), default_float()) / order
    op = 501
    I = tf.cast(GetAllListPairs(op, 1), default_float()) / (op-1)
    #print(I)
    B = BernsteinPolynomial(order)
    BX = B(I)
    #print(BX.shape)
    P = tf.linalg.pinv( tf.squeeze(BX ** 2, axis = 1), rcond = 0.01)
    #print(P)
    solve = tf.matmul(P, sigma2N / muN ** 2 * tf.ones([op, 1], dtype = default_float()))  
    #print(solve)
    #solve = tf.math.abs(solve)
    #return tf.math.sqrt(tf.math.log(1 + solve/muN**2))
    #a = 1000.0
    return tf.math.sqrt(tf.math.log(1.0 + solve))


if __name__ == '__main__':
    out = prior_adjusting(100)
    out2 = prior_adjusting(10)
    #print(out)
    pass
