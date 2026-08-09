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
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)

        du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx
        du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx

        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
        du_dy[0, :] = (u[1, :] - u[0, :]) / dy
        du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy

        return np.stack((du_dx, du_dy), axis=-1)
    else:

        du_dx = torch.zeros_like(u)
        du_dy = torch.zeros_like(u)

        du_dx[..., :, 1:-1] = (u[..., :, 2:] - u[..., :, :-2]) / (2 * dx)
        du_dx[..., :, 0] = (u[..., :, 1] - u[..., :, 0]) / dx
        du_dx[..., :, -1] = (u[..., :, -1] - u[..., :, -2]) / dx

        du_dy[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2 * dy)
        du_dy[..., 0, :] = (u[..., 1, :] - u[..., 0, :]) / dy
        du_dy[..., -1, :] = (u[..., -1, :] - u[..., -2, :]) / dy

        return torch.stack((du_dx, du_dy), dim=-1)

def divergence(u, dx, dy, boundary = None, numpy_=False):
    grad_u = grad(u=u,
                  dx=dx,
                  dy=dy,
                  boundary=boundary,
                  numpy_=numpy_)
    
    du_dx, du_dy = grad_u[..., 0], grad_u[..., 1]


    if numpy_:
        dux_dx = du_dx[..., 0]
        duy_dy = du_dy[..., 1]
    else:
        dux_dx = du_dx[:, 0]
        duy_dy = du_dy[:, 1]

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
        # u_x, u_y = u[1:-1, 1:-1, 0], u[1:-1, 1:-1, 1]
        u_x, u_y = u[..., 0], u[..., 1]
        u_y = u_y[..., np.newaxis]
        u_x = u_x[..., np.newaxis]
    else:
        du_dx, du_dy = grad_u[..., 0], grad_u[..., 1]
        # u_x, u_y = u[:, 0, 1:-1, 1:-1], u[:, 1, 1:-1, 1:-1]
        u_x, u_y = u[:, 0], u[:, 1]
        u_y = u_y.unsqueeze(1)
        u_x = u_x.unsqueeze(1)

    conv = u_x * du_dx + u_y * du_dy

    return conv

def laplacian(u, dx, dy, boundary=None, numpy_=False):
    if numpy_:
        d2u_dx2 = np.zeros_like(u)
        d2u_dy2 = np.zeros_like(u)

        d2u_dx2[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / (dx**2)
        d2u_dx2[:, 0] = (u[:, 0] - 2*u[:, 1] + u[:, 2]) / (dx**2)
        d2u_dx2[:, -1] = (u[:, -1] - 2*u[:, -2] + u[:, -3]) / (dx**2)

        d2u_dy2[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / (dy**2)
        d2u_dy2[0, :] = (u[0, :] - 2*u[1, :] + u[2, :]) / (dy**2)
        d2u_dy2[-1, :] = (u[-1, :] - 2*u[-2, :] + u[-3, :]) / (dy**2)

        return d2u_dx2 + d2u_dy2
    else:
        if u.shape[1] == 2:
            d2u_dx2 = torch.zeros_like(u)
            d2u_dy2 = torch.zeros_like(u)

            d2u_dx2[..., 1:-1] = (u[..., 2:] - 2*u[..., 1:-1] + u[..., :-2]) / (dx**2)
            d2u_dx2[..., 0] = (u[..., 0] - 2*u[..., 1] + u[..., 2]) / (dx**2)
            d2u_dx2[..., -1] = (u[..., -1] - 2*u[..., -2] + u[..., -3]) / (dx**2)

            d2u_dy2[..., 1:-1, :] = (u[..., 2:, :] - 2*u[..., 1:-1, :] + u[..., :-2, :]) / (dy**2)
            d2u_dy2[..., 0, :] = (u[..., 0, :] - 2*u[..., 1, :] + u[..., 2, :]) / (dy**2)
            d2u_dy2[..., -1, :] = (u[..., -1, :] - 2*u[..., -2, :] + u[..., -3, :]) / (dy**2)

            return d2u_dx2 + d2u_dy2

        u = torch.nn.functional.pad(u, (1, 1, 1, 1), mode="constant", value=0)

        d2u_dx2 = (torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)) / (dx ** 2)
        d2u_dy2 = (torch.roll(u, -1, dims=-2) - 2 * u + torch.roll(u, 1, dims=-2)) / (dy ** 2)
        
        return (d2u_dx2 + d2u_dy2)[..., 1:-1, 1:-1]