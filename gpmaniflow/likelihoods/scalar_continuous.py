import tensorflow as tf

from gpflow.likelihoods.base import ScalarLikelihood

class Nakagami(ScalarLikelihood):
    '''
    The Nakagami likelihood is differing from the GPflow standard, as it does not necessarily have take a GP as its input. 
    It is, so far, only used for the observation model when distances are considered.
    '''
    def __init__(self, lcl = 0, rcl = float('inf'), **kwargs):
        super().__init__(**kwargs)
        self.lcl = lcl
        self.rcl = rcl
        '''
        lcl and rcl are left and right censoring limits respectively.
        '''
        
    def _scalar_log_prob(self, m_and_Omega, Y):
        def log_tail(x, m_and_Omega):
            m = m_and_Omega[0]
            Omega = m_and_Omega[1]
            return tf.math.log(1 - tf.math.igamma(m, m/Omega * x ** 2)) # + 1e-8
        def logpdf(x, m_and_Omega):
            m = m_and_Omega[0]
            Omega = m_and_Omega[1]
            return - tf.math.lgamma(m) - m * tf.math.log(Omega/m) + 2*m - tf.math.log(x) - tf.square(x) * m / Omega
        
        func = lambda a: tf.cond(tf.math.less(a[0],self.rcl), lambda: logpdf(a[0], a[1]), lambda: log_tail(self.rcl, a[1])) # This only considers right-censoring.
        res = tf.map_fn(func, (Y, tf.transpose(m_and_Omega)), dtype = Y.dtype)
        return res