import numpy as np
import torch

def unpack_grad(grad_u, numpy_=False):
    if numpy_:
        du_dy, du_dx = grad_u[..., 0], grad_u[..., 1]

        duy_dy, dux_dy = du_dy[..., 0], du_dy[..., 1]
        duy_dx, dux_dx = du_dx[..., 0], du_dx[..., 1]
    else:
        du_dx, du_dy = grad_u[:, 0], grad_u[:, 1]

        dux_dy, duy_dy = du_dy[:, 0], du_dy[:, 1]
        dux_dx, duy_dx = du_dx[:, 0], du_dx[:, 1]

    return duy_dy, dux_dy, duy_dx, dux_dx


def grad(u, dx, dy, boundary=None, numpy_=False):
    if numpy_:
        du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
        du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)

        du_dx = du_dx[1:-1, 1:-1]
        du_dy = du_dy[1:-1, 1:-1]

        return np.stack((du_dx, du_dy), axis=-1)
    else:
        # u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), mode="constant", value=0)
        u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), mode="replicate")
        # u_padded = u
        du_dx = (torch.roll(u_padded, -1, dims=-1) - torch.roll(u_padded, 1, dims=-1)) / (2 * dx)
        du_dy = (torch.roll(u_padded, -1, dims=-2) - torch.roll(u_padded, 1, dims=-2)) / (2 * dy)

        du_dx = du_dx[..., 1:-1, 1:-1]
        du_dy = du_dy[..., 1:-1, 1:-1]

        return torch.stack((du_dx, du_dy), dim=1)

def divergence(u, dx, dy, boundary = None, numpy_=False):
    grad_u = grad(u=u,
                  dx=dx,
                  dy=dy,
                  boundary=boundary,
                  numpy_=numpy_)
    
    du_dx, du_dy = grad_u[..., 0], grad_u[..., 1]

    dux_dx, duy_dx = du_dx[..., 0], du_dx[..., 1]
    dux_dy, duy_dy = du_dy[..., 0], du_dy[..., 1]

    # duy_dy, dux_dy, duy_dx, dux_dx = unpack_grad(grad_u, numpy_)

    div = dux_dx + duy_dy

    return div

def convective(u, dx, dy, boundary = None, numpy_=False):
    grad_u = grad(u=u,
                  dx=dx,
                  dy=dy,
                  boundary=boundary,
                  numpy_=numpy_)

    if numpy_:
        du_dx, du_dy = grad_u[..., 0], grad_u[..., 1]
        # u_y, u_x = u[1:-1, 1:-1, 0], u[1:-1, 1:-1, 1]
        u_x, u_y = u[1:-1, 1:-1, 0], u[1:-1, 1:-1, 1]
        u_y = u_y[..., np.newaxis]
        u_x = u_x[..., np.newaxis]
    else:
        du_dx, du_dy = grad_u[:, 0], grad_u[:, 1]
        u_x, u_y = u[:, 0], u[:, 1]
        u_y = u_y.unsqueeze(1)
        u_x = u_x.unsqueeze(1)

    conv = u_x * du_dx + u_y * du_dy

    # conv = u[..., 0] * du_dx + u[..., 1] * du_dy

    return conv

# def convective(u, dx, dy, boundary = None, numpy_=False):
#     grad_u = grad(u=u,
#                   dx=dx,
#                   dy=dy,
#                   boundary=boundary,
#                   numpy_=numpy_)

#     dux_dx = grad_u[..., 0, 0]
#     dux_dy = grad_u[..., 0, 1]
#     duy_dx = grad_u[..., 1, 0]
#     duy_dy = grad_u[..., 1, 1]

#     u_x = u[1:-1, 1:-1, 0]
#     u_y = u[1:-1, 1:-1, 1]

#     conv = np.zeros_like(u[1:-1, 1:-1, :])

#     conv[..., 0] = u_x * dux_dx + u_y * dux_dy
#     conv[..., 1] = u_x * duy_dx + u_y * duy_dy

#     return conv


def laplacian(u, dx, dy, boundary=None, numpy_=False):
    if numpy_:

        d2u_dx2 = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / (dx ** 2)
        d2u_dy2 = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / (dy ** 2)

        res = (d2u_dx2 + d2u_dy2)[1:-1, 1:-1]

        return res
    else:
        u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), mode="constant", value=0)
        # u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), mode="replicate")
        # u_padded = u

        d2u_dx2 = (torch.roll(u_padded, -1, dims=-1) - 2 * u_padded + torch.roll(u_padded, 1, dims=-1)) / (dx ** 2)
        d2u_dy2 = (torch.roll(u_padded, -1, dims=-2) - 2 * u_padded + torch.roll(u_padded, 1, dims=-2)) / (dy ** 2)
        
        res = (d2u_dx2 + d2u_dy2)[..., 1:-1, 1:-1]
        
        return res
