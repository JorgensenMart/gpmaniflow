import numpy as np
import tensorflow as tf

from .. import logdensities
from ..base import Parameter
from ..utilities import positive
from .base import ScalarLikelihood
from .utils import inv_probit

class Nakagami(ScalarLikelihood):
    def __init__(self, censoring = [None, None], **kwargs):
        super().__init__(**kwargs)
        self.censoring = censoring
        
    def _scalar_log_prob(self, m_and_Omega, Y):
        