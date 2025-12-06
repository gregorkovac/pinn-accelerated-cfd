# import numpy as np
import numpy as np

# def apply_dirichlet_bc(u, idx, val, mask=(1, 1)):
#     """
#     Applies a Dirichlet boundary condition to a scalar or vector field

#     Args:
#         u (numpy.ndarray) - scalar or vector field of shape (n, m) or (n, m, 2)
#         idx (tuple) - index of where to apply the boundary condition
#         val (float) - boundary condition value

#     Returns:
#         (numpy.ndarray) - u with applied boundary condition
#     """

#     if u is None:
#         return None

#     if u.ndim == 2 or len(np.shape(idx)) == 2:
#         u[idx] = val
#         return u

#     # print(np.shape(u))
#     # print(idx)
#     # print(idx + (0,))
#     # exit(0)

#     if mask[0] == 1:
#         u[idx[0], idx[1], 0] = val[0]

#     if mask[1] == 1:
#         u[idx[0], idx[1], 0] = val[1]

#     return u

def apply_dirichlet_bc(u, idx, val, mask=(1, 1)):
    if u is None:
        return None

    if u.ndim == 2:
        u[idx] = val
        return u

    if isinstance(idx, tuple):  
        if mask[0] == 1:
            u[idx[0], idx[1], 0] = val[0]
        if mask[1] == 1:
            u[idx[0], idx[1], 1] = val[1]
    else:
        if mask[0] == 1:
            u[idx, 0] = val[0]
        if mask[1] == 1:
            u[idx, 1] = val[1]

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

    if g["type"] == "cylinder":
        x, y, r = g["pos"]["x"], g["pos"]["y"], g["size"]

        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        mask = (xx - x)**2 + (yy - y)**2 <= r**2

        shifted_mask = (xx - x)**2 + (yy - y)**2 <= (r + 1)**2

        neumann_mask = np.logical_and(shifted_mask, ~mask)

        return np.where(neumann_mask)
        

    x, y = g["pos"]["x"], g["pos"]["y"]
    w, h = g["size"]["width"], g["size"]["height"]
    if "dir" in g.keys():
        dir_x, dir_y = g["dir"]["x"], g["dir"]["y"]
    elif "vel" in g.keys():
        dir_x, dir_y = g["vel"]["x"], g["vel"]["y"]
        if dir_x != 0:
            dir_x = 1
        if dir_y != 0:
            dir_y = 1
    
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

def get_normal(g):
    if "dir" in g.keys():
        dir_x, dir_y = g["dir"]["x"], g["dir"]["y"]
    elif "vel" in g.keys():
        dir_x, dir_y = g["vel"]["x"], g["vel"]["y"]
        if dir_x != 0:
            dir_x = 1
        if dir_y != 0:
            dir_y = 1

    return np.array([dir_y, dir_x])

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

    
    # Get width and height and perform padding
    if u is not None:
        u = np.pad(u, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        height, width = np.shape(u)[0], np.shape(u)[1]

    if p is not None:
        p = np.pad(p, pad_width=1, mode='constant', constant_values=0)
        height, width = np.shape(p)[0], np.shape(p)[1]

    # Iterate over geometries
    for g in geometry:

        # Get the boudnary condition index
        idx = get_index(g, height, width)

        # Apply boundary conditions depending on the type
        if g["type"] == "wall":
            neumann_idx = get_neumann_index(g, height, width)
            normal = get_normal(g)
            u = apply_dirichlet_bc(u, idx, (0, 0))# mask=(1 - np.abs(normal)))
            p = apply_neumann_bc(p, idx, neumann_idx)
        elif g["type"] == "inflow":
            neumann_idx = get_neumann_index(g, height, width)

            # if u is not None:
            #     for i in range(1, 62):
            #         r = i - 62 / 2
            #         R = 62
            #         u[i, 0] = np.array([g["vel"]["x"], g["vel"]["y"]]) * (1 - (r / R) ** 2)

            u = apply_dirichlet_bc(u, idx, (g["vel"]["x"],
                                            g["vel"]["y"]))

            p = apply_neumann_bc(p, idx, neumann_idx)
            # p = apply_dirichlet_bc(p, idx, 1)
        elif g["type"] == "outflow":
            neumann_idx = get_neumann_index(g, height, width)
            u = apply_neumann_bc(u, idx, neumann_idx)
            p = apply_dirichlet_bc(p, idx, 0)
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

    im = np.zeros((height + 2, width + 2))

    for g in geometry:
        idx = get_index(g, height + 2, width + 2)
        im[idx] = 1

    im = im[1:-1, 1:-1]

    return im