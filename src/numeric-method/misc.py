import numpy as np

from derivatives import *
from geometry import get_boundary_image

def navier_stokes_error(u, p, density, viscosity, time_step, grid_spacing, u_prev = None, u_next = None, geometry = None):
    """
    Computes the errors for momentum and divergence

    Args:
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        p (numpy.ndarray) - pressure scalar field of shape (n, m)
        density (float) - density constant
        viscosity (float) - viscosity constant
        time_step (float) - time step
        grid_spacing (float) - grid spacing
        u_prev (numpy.ndarray) - velocity vector field of shape (n, m, 2) in previous time step
        u_next (numpy.ndarray) - velocity vector field of shape (n, m, 2) in next time step
        geometry (dict) - geometry dictionary
    
    Returns:
        (tuple) - error stats of shape (divergence error mean, divergence error standard deviation, 
                                        momentum error mean, momentum error standard deviation)
    """ 

    # Compute the time derivative
    if u_prev is not None and u_next is not None:
        du_dt = (u_next - u_prev) / (2 * time_step)
    elif u_prev is not None:
        du_dt = (u - u_prev) / time_step
    elif u_next is not None:
        du_dt = (u_next - u) / time_step
    else:
        raise RuntimeError("Navier-Stokes error: could not compute time derivative")
    
    # Compute errors
    divergence_err = del_dot_v(u, grid_spacing)
    momentum_err = du_dt + v_dot_del_v(u, grid_spacing) + 1/density * del_v(p, grid_spacing) - viscosity * laplace(u, grid_spacing)

    # print(f"sum(du_dt ** 2) = {np.sum(du_dt ** 2)}")
    # print(f"sum(u dot del u ** 2) = {np.sum(v_dot_del_v(u, grid_spacing) ** 2)}")
    # print(f"sum(1/density * del_v(p, grid_spacing) ** 2) = {np.sum((1/density * del_v(p, grid_spacing)) ** 2)}")
    # print(f"sum(- viscosity * laplace(u, grid_spacing) ** 2) = {np.sum((- viscosity * laplace(u, grid_spacing)) ** 2)}")
    # print()

    # Remove boundaries
    if geometry is None:
        divergence_err = divergence_err[1:-1, 1:-1]
        momentum_err = momentum_err[1:-1, 1:-1]
    else:
        width, height = np.shape(u)[1], np.shape(u)[0]
        boundary_image = get_boundary_image(geometry, width, height)
        
        momentum_err[boundary_image == 1] = 0
        divergence_err[boundary_image == 1] = 0

        # print(np.sum(boundary_image) / (np.shape(boundary_image)[0] * np.shape(boundary_image)[1]))

    # return np.mean(divergence_err), np.std(divergence_err), np.mean(momentum_err), np.std(momentum_err)
    return np.sum(divergence_err ** 2), np.std(divergence_err), np.sum(momentum_err ** 2), np.std(momentum_err)