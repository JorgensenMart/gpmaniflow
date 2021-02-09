import numpy as np
 
def factorial(n):
    return np.prod(range(1,n+1))

def bezier_coef(n):
    k = np.ones(n+1)
    nk = np.ones(n+1)
    for i in range(n+1):
        k[i] = factorial(i)
        nk[i] = factorial(n-i)
    return factorial(n) / (k*nk)