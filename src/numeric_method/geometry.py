# import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import copy

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
    
    if isinstance(idx, np.ndarray) and idx.dtype == bool:
        i_b, j_b = np.where(idx)
        i_in = neumann_idx[i_b, j_b, 0]
        j_in = neumann_idx[i_b, j_b, 1]

        i_vec = i_in - i_b
        j_vec = j_in - j_b

        if u.ndim == 2:
            # u[i_b, j_b] = u[i_in, j_in]
            u[i_b - i_vec, j_b - j_vec] = u[i_b + i_vec, j_b + j_vec]
        else:
            # u[i_b, j_b, :] = u[i_in, j_in, :]
            u[i_b - i_vec, j_b - j_vec, :] = u[i_b + i_vec, j_b + j_vec, :]

        return u
    
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

        dist = np.sqrt((xx - x)**2 + (yy - y)**2)

        boundary_mask = np.logical_and(dist <= r + 0.5, dist >= r - 0.5)

        i_b, j_b = np.where(boundary_mask)

        nx = (j_b - x) / dist[i_b, j_b]
        ny = (i_b - y) / dist[i_b, j_b]

        i_in = np.round(i_b - ny).astype(int)
        j_in = np.round(j_b - nx).astype(int)

        nidx = np.stack((yy, xx), axis=-1)

        nidx[boundary_mask, :] = np.stack((i_in, j_in), axis=1)

        return nidx

        # print((i_b, j_b), (i_in, j_in))

        # return (i_b, j_b), (i_in, j_in)

        # mask = (xx - x)**2 + (yy - y)**2 <= r**2

        # shifted_mask = (xx - x)**2 + (yy - y)**2 <= (r + 1)**2

        # neumann_mask = np.logical_and(shifted_mask, ~mask)

        # return np.where(neumann_mask)
        

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

    # return (slice(y + dir_y, y + h + dir_y), slice(x + dir_x, x + w + dir_x))


    # return [(slice(y + dir_y, y + h + dir_y), 
    #         slice(x + dir_x, x + w + dir_x)),
    #         (slice(y - dir_y, y + h - dir_y), 
    #         slice(x - dir_x, x + w - dir_x))]

    # return [(slice(y + dir_y, y + h + dir_y), 
    #             slice(x + dir_x, x + w + dir_x)),
    #             (slice(y - dir_y, y + h), 
    #             slice(x - dir_x, x + w))]

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
    # if u is not None:
    #     u = np.pad(u, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    #     height, width = np.shape(u)[0], np.shape(u)[1]

    # if p is not None:
    #     p = np.pad(p, pad_width=1, mode='constant', constant_values=0)
    #     height, width = np.shape(p)[0], np.shape(p)[1]

    if u is not None:
        height, width = np.shape(u)[0], np.shape(u)[1]

    if p is not None:
        height, width = np.shape(p)[0], np.shape(p)[1]

    # if p is not None:
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(p)
    #     ax[0].set_title("b4")

    # Iterate over geometries
    for g in geometry:

        # Get the boudnary condition index
        idx = get_index(g, height, width)
        
        # g_inner = copy.deepcopy(g)
        # g_inner["pos"]["x"] += 1
        # g_inner["pos"]["y"] += 1
        # inner_idx = get_index(g_inner, height - 2, width - 2)

        # Apply boundary conditions depending on the type
        if g["type"] == "wall":
            neumann_idx = get_neumann_index(g, height, width)
            normal = get_normal(g)
            # print(neumann_idx)
            u = apply_dirichlet_bc(u, idx, (0, 0))# mask=(1 - np.abs(normal)))
            # p = apply_neumann_bc(p, idx, neumann_idx)
        elif g["type"] == "inflow":
            neumann_idx = get_neumann_index(g, height, width)

            u = apply_dirichlet_bc(u, idx, (g["vel"]["x"],
                                            g["vel"]["y"]))
            # p = apply_neumann_bc(p, idx, neumann_idx)

            # p = apply_dirichlet_bc(p, idx, 1)
        elif g["type"] == "outflow":
            neumann_idx = get_neumann_index(g, height, width)
            u = apply_neumann_bc(u, idx, neumann_idx)
            p = apply_dirichlet_bc(p, idx, 0)
        elif g["type"] == "cylinder":
            neumann_idx = get_neumann_index(g, height, width)

            # inner_idx = get_index(g, height, width)

            u = apply_dirichlet_bc(u, idx, (0, 0))
            # p = apply_neumann_bc(p, idx, neumann_idx)

    # if u is not None and p is not None:
    #     fig, ax = plt.subplots(1, 3)
    #     ax[0].imshow(u[..., 0])
    #     ax[1].imshow(u[..., 1])
    #     ax[2].imshow(p)
    #     plt.show()
    # if p is not None:
    #     ax[1].imshow(p)
    #     ax[1].set_title("after")
    #     plt.show() 

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

    # im = im[1:-1, 1:-1]

    return im