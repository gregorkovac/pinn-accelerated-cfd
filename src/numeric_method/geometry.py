import numpy as np

def apply_dirichlet_bc(u, idx, val):
    """
    Applies a Dirichlet boundary condition to a scalar or vector field

    Args:
        u (numpy.ndarray) - scalar or vector field of shape (n, m) or (n, m, 2)
        idx (tuple) - index of where to apply the boundary condition
        val (float) - boundary condition value

    Returns:
        (numpy.ndarray) - u with applied boundary condition
    """

    if u is None:
        return None
    
    u[idx] = val
    return u

def apply_neumann_bc(u, idx, neumann_idx):
    """
    Applies a Neumann boundary condition to a scalar or vector field

    Args:
        u (numpy.ndarray) - scalar or vector field of shape (n, m) or (n, m, 2)
        idx (tuple) - index of where to apply the boundary condition
        neumann_idx (tuple) - index of which part of u to copy to the index to have a zero derivative
    
    Returns:
        (numpy.ndarray) - u with applied boundary condition
    """

    if u is None:
        return None

    u[idx] = u[neumann_idx]

    return u

def get_index(g, height = None, width = None):
    """
    Returns the index of a boundary condition

    Args:
        g (dict) - geometry dictionary
        height (int) - height in number of points
        width (int) - width in number of points

    Returns:
        (tuple) - index of the boundary condition
    """

    # Special case for cylinder
    if g["type"] == "cylinder":

        # Create a grid
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        # Compute the distances from the cylinder center
        distance_from_center = np.sqrt((x - g["pos"]["x"])**2 + (y - g["pos"]["y"])**2)

        # Create the mask
        mask = distance_from_center <= g["size"]

        return mask
    
    x, y = g["pos"]["x"], g["pos"]["y"]
    w, h = g["size"]["width"], g["size"]["height"]

    if x == -1:
        x = width - 1
    if y == -1:
        y = height - 1
    if w == -1:
        w = width
    if h == -1:
        h = height

    return (slice(y, y + h), 
            slice(x, x + w))

def get_neumann_index(g, height = None, width = None):
    """
    Computes the 'shifted' index used for Neumann boundary conditions

    Args:
        g (dict) - geometry dictionary
        height (int) - height in number of points
        width (int) - width in number of points

    Returns:
        (tuple) - the shifted index
    """

    x, y = g["pos"]["x"], g["pos"]["y"]
    w, h = g["size"]["width"], g["size"]["height"]
    dir_x, dir_y = g["dir"]["x"], g["dir"]["y"]
    
    if x == -1:
        x = width - 1
    if y == -1:
        y = height - 1
    if w == -1:
        w = width
    if h == -1:
        h = height

    return (slice(y + dir_y, y + h + dir_y), 
            slice(x + dir_x, x + w + dir_x))

def boundary(geometry, u = None, p = None):
    """
    Applies boundary conditions for a given geometry

    Args:
        g (dict) - geometry dictionary
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        p (numpy.ndarray) - pressure scalar field of shape (n, m)

    Returns:
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2) with applied boundary conditions
        p (numpy.ndarray) - pressure scalar field of shape (n, m) with applied boundary conditions
    """
        
    # Get width and height
    if u is not None:
        height, width = np.shape(u)[0], np.shape(u)[1]
    elif p is not None:
        height, width = np.shape(p)[0], np.shape(p)[1]
    else:
        raise RuntimeError("u or p is needed to compute boundary")

    # Iterate over geometries
    for g in geometry:

        # Get the boudnary condition index
        idx = get_index(g, height, width)

        # Apply boundary conditions depending on the type
        if g["type"] == "wall":
            u = apply_dirichlet_bc(u, idx, (0, 0))
        elif g["type"] == "inflow":
            u = apply_dirichlet_bc(u, idx, (g["vel"]["x"],
                                            g["vel"]["y"]))
        elif g["type"] == "outflow":
            neumann_idx = get_neumann_index(g, height, width)

            u = apply_neumann_bc(u, idx, neumann_idx)
            p = apply_neumann_bc(p, idx, neumann_idx)
            # p = apply_dirichlet_bc(p, idx, 0)
        elif g["type"] == "cylinder":
            u = apply_dirichlet_bc(u, idx, (0, 0))

    return u, p

def get_boundary_image(geometry, width, height):
    """
    Returns a binary image in which there are zeros on the boundary

    Args:
        geometry (dict) - geometry dictionary
        width (int) - width in number of points
        height (int) - height in number of points

    Returns:
        im (numpy.ndarray) - binary image of shape (height, width)
    """

    im = np.zeros((height, width))

    for g in geometry:
        idx = get_index(g, height, width)
        im[idx] = 1

    return im