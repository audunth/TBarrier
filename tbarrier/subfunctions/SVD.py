import numpy as np
from math import sqrt, cos, sin, atan2

def SVD(gradFmap):
    '''Compute the singular value decomposition. For an arbitrary matrix F, decomposite as F = P \Sigma Q^T. 
    
    Parameters:
        gradFmap: array (2,2) arbitrary 2 by 2 matrix
    
    Returns:
        P: orthogonal rotation tensor
        Sig: array(2,2) diagonal matrix of singular values
        Q: orthogonal rotation tensor
    '''
    Su = gradFmap@gradFmap.transpose()
    theta = 0.5*atan2(Su[0,1]+Su[1,0], Su[0,0]-Su[1,1])
    Ctheta = cos(theta)
    Stheta = sin(theta)
    P = np.array([[Ctheta, -Stheta], [Stheta, Ctheta]])

    Sw = gradFmap.transpose()@gradFmap
    phi = 0.5*atan2((Sw[0,1]+Sw[1,0]), (Sw[0,0]-Sw[1,1]))
    Cphi = cos(phi)
    Sphi = sin(phi)
    W = np.array([[Cphi, -Sphi], [Sphi, Cphi]])

    SUsum= Su[0,0]+Su[1,1]
    SUdif= sqrt((Su[0,0]-Su[1,1])**2 + 4*(Su[0,1]*Su[1,0]))
    
    if SUsum-SUdif < 0: # This happens due to numerical inaccuracies
        svals = np.array([sqrt((SUsum+SUdif)/2), 0])
    
    else:
        svals= np.array([sqrt((SUsum+SUdif)/2), sqrt((SUsum-SUdif)/2)])
   
    SIG = np.diag(svals)

    S = P.transpose()@gradFmap@W
    C = np.diag([np.sign(S[0,0]), np.sign(S[1,1])])
    Q = W@C
    
    return P, SIG, Q
