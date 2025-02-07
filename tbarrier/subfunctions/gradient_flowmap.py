import numpy as np
from numba import njit, prange
import tbarrier

def gradient_flowmap(time, x, X, Y, Interpolant_u, Interpolant_v, periodic, defined_domain, bool_unsteady, time_data, aux_grid, verbose=False):
    '''
    Calculates the gradient of the flowmap for a flow given by u/v velocities, starting from a set of given initial conditions (ICs). 
    The ICs can be specified as an array. 
    
    Parameters:
        time: array (Nt,),  time instant  
        x: array (2, Npoints),  array of ICs
        X: array (NY, NX)  X-meshgrid
        Y: array (NY, NX)  Y-meshgrid 
        Interpolant_u: Interpolant object for u(x, t)
        Interpolant_v: Interpolant object for v(x, t)
        periodic: list of 3 bools, periodic[i] is True if the flow is periodic in the ith coordinate. Time is i=3.
        bool_unsteady:  specifies if velocity field is unsteady/steady
        time_data: array(Nt, ) time of velocity data
        aux_grid: array(2,), grid spacing for the auxiliary grid
        verbose: bool, if True, function reports progress at every 100th iteration

    Returns:
        gradFmap: array(Nt, 2, 2, Npoints), gradient of the flowmap (2 by 2 matrix) for each time instant and each spatial point
    '''
    # define auxiliary grid spacing
    rho_x = aux_grid[0]
    rho_y = aux_grid[1]
    
    x = x.reshape(2, -1)
    
    X0, XL, XR, XU, XD = [], [], [], [], []

    nan_mask = []
    
    for i in range(x.shape[1]):
        
        xr = x[0, i] + rho_x # float
        xl = x[0, i] - rho_x # float
        yu = x[1, i] + rho_y # float
        yd = x[1, i] - rho_y # float
        
        bool_xr = (tbarrier.check_location(X, Y, defined_domain, np.array([xr, x[1, i]]))[0] == "IN") # bool
        bool_xl = (tbarrier.check_location(X, Y, defined_domain, np.array([xl, x[1, i]]))[0] == "IN") # bool
        bool_yu = (tbarrier.check_location(X, Y, defined_domain, np.array([x[0, i], yu]))[0] == "IN") # bool
        bool_yd = (tbarrier.check_location(X, Y, defined_domain, np.array([x[0, i], yd]))[0] == "IN") # bool
        
        # check initial location of particles. Only compute gradient of flow map for those particles whose auxiliary trajectories are all within the defined flow domain
        if bool_xr and bool_xl and bool_yu and bool_yd:
            
            nan_mask.append(True)
            
        else:
            
            nan_mask.append(False)
    
        X0.append([x[0, i], x[1, i]])
        XL.append([xl, x[1, i]])
        XR.append([xr, x[1, i]])
        XU.append([x[0, i], yu])
        XD.append([x[0, i], yd])
    
    X0 = np.array(X0).transpose()
    XL = np.array(XL).transpose()
    XR = np.array(XR).transpose()
    XU = np.array(XU).transpose()
    XD = np.array(XD).transpose()
    
    # launch trajectories from auxiliary grid
    XLend = tbarrier.integration_dFdt(time, XL, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, verbose)[0] # array (Nt, 2)
    
    XRend = tbarrier.integration_dFdt(time, XR, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, verbose)[0] # array (Nt, 2)
    
    XDend = tbarrier.integration_dFdt(time, XD, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, verbose)[0] # array (Nt, 2)
    
    XUend = tbarrier.integration_dFdt(time, XU, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, verbose)[0] # array (Nt, 2)
    
    # compute gradient of flow map over time interval
    gradFmap = iterate_gradient(XRend, XLend, XUend, XDend) # array (Nt, 2, 2, Nx*Ny)
    
    return gradFmap

@njit(parallel = True)
def iterate_gradient(XRend, XLend, XUend, XDend):
    '''
    Compute gradient of flow map
    '''
    
    gradFmap = np.zeros((XLend.shape[0], 2, 2, XLend.shape[2])) # array (Nt, 2, 2, Nx*Ny)
    
    for i in prange(XLend.shape[2]):      
            
        for j in prange(XLend.shape[0]):

            gradFmap[j,0,0,i] = (XRend[j,0,i]-XLend[j,0,i])/(XRend[0,0,i]-XLend[0,0,i])
            gradFmap[j,1,0,i] = (XRend[j,1,i]-XLend[j,1,i])/(XRend[0,0,i]-XLend[0,0,i])
        
            gradFmap[j,0,1,i] = (XUend[j,0,i]-XDend[j,0,i])/(XUend[0,1,i]-XDend[0,1,i])
            gradFmap[j,1,1,i] = (XUend[j,1,i]-XDend[j,1,i])/(XUend[0,1,i]-XDend[0,1,i])
            
    return gradFmap