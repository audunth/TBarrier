import numpy as np
from math import sqrt

def eigen(A):
    '''The function computes the eigenvalues and eigenvectors of a two-dimensional symmetric matrix.
    
    Parameters:
        A: array(2,2), input matrix 
        
        
    Returns:
        lambda_min: float, minimal eigenvalue
        lambda_max: float, maximal eigenvalue
        v_min: array(2,), minimal eigenvector
        v_max: array(2,), maximal eigenvector
    '''
    A11 = A[0, 0] # float
    A12 = A[0, 1] # float
    A22 = A[1, 1] # float
    
    discriminant = (A11+A22)**2/4-(A11*A22-A12**2) # float
    
    if discriminant < 0 or np.isnan(discriminant):
        return np.nan, np.nan, np.zeros((1,2))*np.nan, np.zeros((1,2))*np.nan
    
    lambda_max = (A11+A22)/2+sqrt(discriminant) # float
    lambda_min = (A11+A22)/2-sqrt(discriminant) # float
    
    v_max = np.array([-A12, A11-lambda_max]) # array (2,)
    v_max = v_max/sqrt(v_max[0]**2+v_max[1]**2) # array (2,)
            
    v_min = np.array([-v_max[1], v_max[0]]) # array (2,)
    
    return lambda_min, lambda_max, v_min, v_max
