
import tensorflow as tf
from gpmaniflow.utils import bezier_coef
from gpflow.config import default_float


class BezierCurve():
    def __init__(self, end_points, order = 1):
        self.end_points = end_points
        self.dimension = end_points[0].shape[1]
        self.order = order
        self.control_points = tf.linspace(end_points[0],end_points[1], self.order + 1)[1:-1] #Potentially make this tf.Variable for training
        self.coeff = tf.constant(bezier_coef(self.order), dtype = default_float()) #Change float
        
    def __call__(self, t):
        ts = tf.expand_dims(t ** 0 * (1-t) ** self.order, 1)
        for i in range(1, self.order + 1): # Can this be done without a for-loop?
            tsnew = tf.expand_dims(t ** i * (1-t) ** (self.order - i) , 1)
            ts = tf.concat(axis = 1, values=[ts, tsnew])
        P = tf.concat(values = [tf.expand_dims(self.end_points[0], axis=0), self.control_points, tf.expand_dims(self.end_points[1], axis=0)], axis = 0)
        P = tf.reshape(P, shape = (P.shape[0], P.shape[2]))
        out = tf.matmul(tf.matmul(ts, tf.linalg.diag(self.coeff)),P)
        return out
    
    def deriv(self, t):
        ''' 
        Returns the derivate dC/dt 
        '''
        ts = tf.expand_dims(t ** 0 * (1-t) ** (self.order - 1), 1)
        for i in range(1, self.order): # Can this be done without a for-loop?
            tsnew = tf.expand_dims(t ** i * (1-t) ** (self.order - 1 - i ), 1)
            ts = tf.concat(axis = 1, values=[ts, tsnew])
        #tsnew = tf.expand_dims(self.order * t ** (self.order - 1),1)
        #ts = tf.concat(axis = 1, values=[ts, tsnew])
        P = tf.concat(values = [tf.expand_dims(self.end_points[0], axis=0), self.control_points, tf.expand_dims(self.end_points[1], axis=0)], axis = 0)
        P = tf.reshape(P, shape = (P.shape[0], P.shape[2]))
        P = P[1:] - P[:-1]
        coeff = tf.constant(bezier_coef(self.order - 1), dtype = default_float())
        out = self.order * tf.matmul(tf.matmul(ts, tf.linalg.diag(coeff)),P)
        return out
