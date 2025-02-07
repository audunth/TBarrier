import numpy as np
import tbarrier

def _FTLE(gradFmap, lenT):
    '''
    Calculate the Finite Time Lyapunov Exponent (FTLE) given the gradient of the flow map over an interval [t_0, t_N].
    
    Parameters:
        gradFmap: array(2, 2), Gradient of flow map (2 by 2 matrix).
        lenT: float, the length of the time-interval |t_N - t_0|
        
    Returns:
        FTLE, float, FTLE value
    '''
        
    # compute maximum singular value of deformation gradient
    sigma_max = tbarrier.SVD(gradFmap)[1][0,0] # float
    
    # If sigma_max < 1, then set to 1. This happens due to numerical inaccuracies or when the flow is compressible.
    # Since we inherently assumed that the flow is incompressible we set sigma_max = 1 if condition is violated.
    if sigma_max < 1:
        return 0
    
    FTLE = 1/(lenT)*np.log(sigma_max) # float
                        
    return FTLE # float


def _FTLE_C(C, lenT):
    '''
    Calculate the Finite Time Lyapunov Exponent (FTLE) given the Cauchy-Green strain tensor over an interval [t_0, t_N].
    
    Parameters:
        C: array(2, 2), Cauchy-Green strain tensor (2 by 2 matrix).
        lenT: float, the length of the time-interval |t_N - t_0|
        
    Returns:
        FTLE, float, FTLE value
    '''
    
    # compute maximum eigenvalue of CG strain tensor
    lambda_max = tbarrier.eigen(C)[1] # float
    
    # If lambda_max < 1, then set to 1. This happens due to numerical inaccuracies or when the flow is compressible.
    # Since we inherently assumed that the flow is incompressible we set sigma_max = 1 if condition is violated.
    if lambda_max < 1:
        return 0
    
    FTLE = 1/(2*lenT)*np.log(lambda_max) # float
    
    return FTLE # float

