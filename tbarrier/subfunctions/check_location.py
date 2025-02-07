import numpy as np
from numba import njit

@njit
def check_location(X, Y, defined_domain, x, no_nans_in_domain = False):
    '''This function evaluates the location of the particle at x. 
    It returns the leftsided indices of the meshgrid X, Y where the particle is located.
    Based on the domain where the flow field is defined, 
    the location of the particle is categorized either as being 
    1. inside the flow domain:"IN"; 
        This happens at points where the velocity field is locally well defined 
        (= The velocities at the four adjacent grid points of the mesh is defined)
    2. outside the flow domain: "OUT"; 
        This happens at points where the velociy field is not defined at all 
        (= The velocities at the four adjacent grid points of the mesh is not defined)
    3. at the boundary: "BOUNDARY"; 
        This happens at points where the velocity field is only partially defined 
        such as at a wall boundary or at the interface between land and sea.
        
    Parameters:
        X: array(Ny, Nx), X-grid
        Y: array(Ny, Nx), Y-grid
        defined_domain: array(Ny, Nx), points in the grid where the velocity is defined
        x: array(2,), position to querry
        no_nans_in_domain: bool, Guarantee that there aren't any nans in the domain. Default is False
        
    Returns:
        loc: "IN", "OUT", "BOUNDARY"
        idx_x: indicate the position if there are nans in the domain
        idx_y: indicate the position if there are nans in the domain
    '''
    # Define boundaries
    Xmax = X[0,-1]
    Xmin = X[0, 0]
    Ymin = Y[0, 0]
    Ymax = Y[-1,0]
    
    # current position
    xp = x[0]
    yp = x[1]
    
    # if there are non nans inside the domain, then we can just worry about the boundaries
    if no_nans_in_domain:
        
        if Xmin < xp < Xmax and Ymin < yp < Ymax:
            
            loc = "IN"
            
            return loc, None, None
        
        else:
        
            loc = "OUT"
            
            return loc, None, None
 
    # if there are nans in the domain (e.g. Land in the ocean), then we need to take that into account
    else:
        
        # compute left/lower indices of location of the particle with respect to the meshgrid
        idx_x = np.searchsorted(X[0,:], xp)
        idx_y = np.searchsorted(Y[:,0], yp)
        
        # check if particle outside rectangular boundaries
        if xp < Xmin or xp > Xmax or yp < Ymin or yp > Ymax or np.isnan(xp) or np.isnan(yp):
            
            loc = "OUT"
        
        else:
            
            # particle at the left boundary of domain
            if idx_x == 0:
                    
                idx_x = 1
            
            # particle at the lower boundary of domain
            if idx_y == 0:
                    
                idx_y = 1
            
            Condition_nan = np.sum(defined_domain[idx_y-1:idx_y+1, idx_x-1:idx_x+1].ravel())
        
            if Condition_nan == 4:
        
                loc = "IN"
                    
            else:
            
                loc = "OUT"
         
        return loc, idx_x, idx_y