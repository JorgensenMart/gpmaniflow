import tensorflow as tf

class BezierProcess(BayesianModel): # Where should we inherit from?
    '''
    Implementation of a Bezier Stochastic Process. 
    '''
    def __init__(
            self,
            likelihood,
            ):
        self.BezierSurface =

    def predict_f(self, Xnew: InputData):
        
        # P should hold all the control points. More accurately the parameters of their distribution.
        # Parameters should be in the first dimension. Output dimension in the second.
        # The remainding should then [O,O,O,...,O] (the number of input dimensions.
        # This means P.shape = [2, D, O, O, ..., O]

        # B should be of size [O,O,...,O] (see above). Each dimension corresponds to a dimension of
        # input space.
        # This means B.shape = [O, O, ...,O]

        return f_mean, f_var
