from scipy.interpolate import RectBivariateSpline as RBS

def interpolant_unsteady(X, Y, U, V, method = "cubic"):
    '''
    Unsteady wrapper for scipy.interpolate.RectBivariateSpline. Creates a list of interpolators for u and v velocities
    
    Parameters:
        X: array (Ny, Nx), X-meshgrid
        Y: array (Ny, Nx), Y-meshgrid
        U: array (Ny, Nx, Nt), U velocity
        V: array (Ny, Nx, Nt), V velocity
        method: Method for interpolation. Default is 'cubic', can be 'linear'
        
    Returns:
        Interpolant: list (2,), U and V  interpolators
    '''
    # Cubic interpolation
    if method == "cubic":
                
        kx = 3
        ky = 3
               
    # linear interpolation
    elif method == "linear":
            
        kx = 1
        ky = 1  
            
    # define u, v interpolants
    Interpolant = [[], []]
                    
    for j in range(U.shape[2]):
                
        Interpolant[0].append(RBS(Y[:,0], X[0,:], U[:,:,j], kx=kx, ky=ky))
        Interpolant[1].append(RBS(Y[:,0], X[0,:], V[:,:,j], kx=kx, ky=ky))
    
    return Interpolant

def interpolant_steady(X, Y, U, V, method = "cubic"):
    '''
    Steady wrapper for scipy.interpolate.RectBivariateSpline. Creates a list of interpolators for u and v velocities
    
    Parameters:
        X: array (Ny, Nx), X-meshgrid
        Y: array (Ny, Nx), Y-meshgrid
        U: array (Ny, Nx), U velocity
        V: array (Ny, Nx), V velocity
        method: Method for interpolation. Default is 'cubic', can be 'linear'
        
    Returns:
        Interpolant: list (2,), U and V  interpolators
    '''
    # Cubic interpolation
    if method == "cubic":
                
        kx = 3
        ky = 3
               
    # linear interpolation
    elif method == "linear":
            
        kx = 1
        ky = 1
            
    # define u, v interpolants
    Interpolant = []
                
    Interpolant.append(RBS(Y[:,0], X[0,:], U, kx=kx, ky=ky))
    Interpolant.append(RBS(Y[:,0], X[0,:], V, kx=kx, ky=ky))  
        
    return Interpolant
