# import numpy as np
import numpy as np
import scipy.ndimage as ndi

def del_v(v, dx, dy):
    """
    Computes gradient of a matrix v of shape (m, n)
    """
    grad = np.gradient(v, dy, dx, edge_order=2)
    sh = np.shape(v)
    grad_v = np.zeros((sh[0], sh[1], 2))
    grad_v[:,:,0] = grad[1]
    grad_v[:,:,1] = grad[0]
    return grad_v

def del_dot_v(v, dx, dy):
    """
    Computes divergence of a vector field v of shape (m, n, 2)
    """

    grad_x = np.gradient(v[:,:,0], dy, dx, edge_order=2)[1]
    grad_y = np.gradient(v[:,:,1], dy, dx, edge_order=2)[0]

    return grad_x + grad_y

def v_dot_del_v(v, dx, dy):
    """
    Computes the product (v dot nabla) v
    for a vector field v of shape (m, n, 2)
    """

    grad_x = np.gradient(v[:,:,0], dy, dx, edge_order=2)
    grad_y = np.gradient(v[:,:,1], dy, dx, edge_order=2)

    res_x = v[:,:,0] * grad_x[1] + v[:,:,1] * grad_x[0]
    res_y = v[:,:,0] * grad_y[1] + v[:,:,1] * grad_y[0]

    res = np.zeros(np.shape(v))

    res[:,:,0] = res_x
    res[:,:,1] = res_y

    return res

def v_dot_del_u(v, u, dx, dy):
    """
    Computes the product (v dot nabla) u
    for vector fields v and u of shape (m, n, 2)
    """

    grad_x = np.gradient(u[:,:,0], dy, dx, edge_order=2)
    grad_y = np.gradient(u[:,:,1], dy, dx, edge_order=2)

    res_x = v[:,:,0] * grad_x[1] + v[:,:,1] * grad_x[0]
    res_y = v[:,:,0] * grad_y[1] + v[:,:,1] * grad_y[0]

    res = np.zeros(np.shape(v))

    res[:,:,0] = res_x
    res[:,:,1] = res_y

    return res

# def laplace(v, dx, dy):
#     if v.ndim == 2:
#         grad = np.gradient(v, dy, dx, edge_order=2)
#         lapl = np.gradient(grad[0], dy, dx, edge_order=2)[0] + np.gradient(grad[1], dy, dx, edge_order=2)[1]
#         return lapl

#     grad_x = np.gradient(v[:,:,0], dy, dx, edge_order=2)
#     grad_y = np.gradient(v[:,:,1], dy, dx, edge_order=2)

#     partial_1 = np.gradient(grad_x[0], dy, dx, edge_order=2)[0]
#     partial_2 = np.gradient(grad_x[1], dy, dx, edge_order=2)[1]
#     partial_3 = np.gradient(grad_y[0], dy, dx, edge_order=2)[0]
#     partial_4 = np.gradient(grad_y[1], dy, dx, edge_order=2)[1]

#     lapl = np.zeros_like(v)
#     lapl[:,:,0] = partial_1 + partial_2
#     lapl[:,:,1] = partial_3 + partial_4

#     return lapl

def laplace(v, dx, dy):
    if v.ndim == 2:
        # Scalar field
        d2v_dx2 = np.gradient(np.gradient(v, dx, axis=1, edge_order=2), dx, axis=1, edge_order=2)
        d2v_dy2 = np.gradient(np.gradient(v, dy, axis=0, edge_order=2), dy, axis=0, edge_order=2)
        return d2v_dx2 + d2v_dy2

    elif v.ndim == 3 and v.shape[2] == 2:
        # Vector field, component-wise Laplacian
        lapl = np.zeros_like(v)
        for i in range(2):
            d2v_dx2 = np.gradient(np.gradient(v[:,:,i], dx, axis=1, edge_order=2), dx, axis=1, edge_order=2)
            d2v_dy2 = np.gradient(np.gradient(v[:,:,i], dy, axis=0, edge_order=2), dy, axis=0, edge_order=2)
            lapl[:,:,i] = d2v_dx2 + d2v_dy2
        return lapl