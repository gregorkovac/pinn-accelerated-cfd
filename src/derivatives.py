import numpy as np
import torch

# def derivative(u, h, boundary, numpy_=False):
#     # Compute forward and backward shifted arrays with Torch/NumPy
#     if numpy_:
#         boundary = boundary.astype(bool)

#         u_fwd = np.roll(u, 1, axis=-1)
#         b_fwd = np.roll(boundary, 1, axis=-1)

#         u_bwd = np.roll(u, -1, axis=-1)
#         b_bwd = np.roll(boundary, -1, axis=-1)

#         if isinstance(h, (int, float)):
#             h_fwd = h
#         else:
#             h_fwd = np.roll(h, 1, axis=-1)
#     else:
#         boundary = boundary.bool()
        
#         u_fwd = torch.roll(u, 1, dims=-1)
#         b_fwd = torch.roll(boundary, 1, dims=-1)

#         u_bwd = torch.roll(u, -1, dims=-1)
#         b_bwd = torch.roll(boundary, -1, dims=-1)

#         if isinstance(h, (int, float)):
#             h_fwd = h
#         else:
#             h_fwd = torch.roll(h, 1, dims=-1)

#     # b_bwd = b_bwd & (1 - b_fwd)

#     b_fwd[boundary] = 0
#     b_bwd[boundary] = 0

#     # Compute central difference mask
#     if numpy_:
#         b_cent = 1 - (b_bwd | b_fwd)
#     else:
#         b_cent = ~ (b_bwd | b_fwd)

#     # if len(np.shape(u)) == 3:
#     #     import matplotlib.pyplot as plt
#     #     fig, ax = plt.subplots(1, 3)
#     #     ax[0].imshow(b_fwd[0])
#     #     ax[1].imshow(b_bwd[0])
#     #     ax[2].imshow(b_cent[0])
#     #     plt.show()

#     # Set edge cases
#     b_fwd[..., 0] = 1
#     b_bwd[..., -1] = 1
#     b_cent[..., 0] = 0
#     b_cent[..., -1] = 0

#     # Compute central, forward, and backward difference
#     diff_cent = (u_bwd - u_fwd) / (h + h_fwd)
#     diff_fwd = (u_bwd - u) / h
#     diff_bwd = (u - u_fwd) / h_fwd

#     # Compute final difference array
#     diff = (b_cent * diff_cent + b_bwd * diff_bwd + b_fwd * diff_fwd)

#     return diff

# def grad(u, dx, dy, boundary, numpy_=False):
#     dx = dx[0].item()
#     dy = dy[0].item()

#     # Prepare transpose versions of the input
#     if numpy_:
#         u_t = np.swapaxes(u, -2, -1)
#         b_t = np.swapaxes(boundary, -2, -1)
#     else:
#         u_t = u.transpose(-2, -1)
#         b_t = boundary.transpose(-2, -1)

#     # Derivative along x axis
#     du_dx = derivative(u=u,
#                        h=dx,
#                        boundary=boundary,
#                        numpy_=numpy_)
    
#     # Derivative of transpose along x axis
#     #  = derivative along y axis
#     du_dy = derivative(u=u_t,
#                        h=dy,
#                        boundary=b_t,
#                        numpy_=numpy_)
    
#     # Tranpose the derivative back
#     if numpy_:
#         du_dy = np.swapaxes(du_dy, -2, -1)
#     else:
#         du_dy = du_dy.transpose(-2, -1)

#     if numpy_:
#         return np.stack([du_dy, du_dx], axis=0)
#     else:
#         return torch.stack([du_dy, du_dx], dim=1)

# def grad(u, dx, dy, boundary=None, numpy_=False):
#     if numpy_:
#         grad_u = np.gradient(u, dy, dx, axis=(0, 1), edge_order=2)
#         return np.stack((grad_u[1], grad_u[0]), axis=-1)
    
#     spacing = (dy, dx) #if isinstance(dy, (int, float)) else (dy[0].item(), dx[0].item())

#     du_dy, du_dx = torch.gradient(u, spacing=spacing, dim=(-1, -2), edge_order=2)

#     return torch.stack((du_dx, du_dy), dim=1)


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

# def unpack_grad(grad_u, numpy_=False):
#     if numpy_:
#         du_dx, du_dy = grad_u[..., 0], grad_u[..., 1]
#         dux_dy, duy_dy = du_dy[..., 0], du_dy[..., 1]
#         dux_dx, duy_dx = du_dx[..., 0], du_dx[..., 1]
#     else:
#         du_dx, du_dy = grad_u[:, 0], grad_u[:, 1]
#         dux_dy, duy_dy = du_dy[:, 0], du_dy[:, 1]
#         dux_dx, duy_dx = du_dx[:, 0], du_dx[:, 1]

#     return duy_dy, dux_dy, duy_dx, dux_dx


def grad(u, dx, dy, boundary=None, numpy_=False):
    if numpy_:
        pad_width = [(1,1), (1,1)] + [(0,0)] * (u.ndim - 2)
        u_padded = np.pad(u, pad_width, mode="edge")
        # u_padded = u

        du_dx = (np.roll(u_padded, -1, axis=1) - np.roll(u_padded, 1, axis=1)) / (2 * dx)
        du_dy = (np.roll(u_padded, -1, axis=0) - np.roll(u_padded, 1, axis=0)) / (2 * dy)

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

    duy_dy, dux_dy, duy_dx, dux_dx = unpack_grad(grad_u, numpy_)

    div = dux_dx + duy_dy

    return div

# def divergence(u, dx, dy, boundary=None, numpy_=False):
#     if numpy_:
#         dux_dx = (np.roll(u[..., 0], -1, axis=1) - np.roll(u[..., 0], 1, axis=1)) / (2 * dx)
#         duy_dy = (np.roll(u[..., 1], -1, axis=0) - np.roll(u[..., 1], 1, axis=0)) / (2 * dy)
#         return dux_dx + duy_dy
#     else:
#         dux_dx = (torch.roll(u[..., 0], -1, dims=2) - torch.roll(u[..., 0], 1, dims=2)) / (2 * dx)
#         duy_dy = (torch.roll(u[..., 1], -1, dims=1) - torch.roll(u[..., 1], 1, dims=1)) / (2 * dy)
#         return dux_dx + duy_dy

def convective(u, dx, dy, boundary = None, numpy_=False):
    grad_u = grad(u=u,
                  dx=dx,
                  dy=dy,
                  boundary=boundary,
                  numpy_=numpy_)

    if numpy_:
        du_dy, du_dx = grad_u[..., 0], grad_u[..., 1]
        u_y, u_x = u[..., 0], u[..., 1]
        u_y = u_y[..., np.newaxis]
        u_x = u_x[..., np.newaxis]
    else:
        du_dy, du_dx = grad_u[:, 0], grad_u[:, 1]
        u_y, u_x = u[:, 0], u[:, 1]
        u_y = u_y.unsqueeze(1)
        u_x = u_x.unsqueeze(1)

    conv = u_x * du_dx + u_y * du_dy

    # conv = u[..., 0] * du_dx + u[..., 1] * du_dy

    return conv

def laplacian(u, dx, dy, boundary=None, numpy_=False):
    if numpy_:
        # u_padded = np.pad(u, (1, 1, 1, 1), mode="edge")
        pad_width = [(1,1), (1,1)] + [(0,0)] * (u.ndim - 2)
        # u_padded = np.pad(u, pad_width, mode="edge")
        u_padded = np.pad(u, pad_width, mode="constant")
        # u_padded = u

        d2u_dx2 = (np.roll(u_padded, -1, axis=1) - 2 * u_padded + np.roll(u_padded, 1, axis=1)) / (dx ** 2)
        d2u_dy2 = (np.roll(u_padded, -1, axis=0) - 2 * u_padded + np.roll(u_padded, 1, axis=0)) / (dy ** 2)

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

# def laplacian(u, dx, dy, boundary = None, numpy_=False):

#     # if numpy_:
#     #     u_padded = np.pad(u, [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
#     # else:
#     #     u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), mode="replicate")

#     #     # dx = dx[0].item()
#     #     # dy = dy[0].item()

#     # d2u_dx2 = (u_padded[..., 1:-1, 2:] - 2 * u_padded[..., 1:-1, 1:-1] + u_padded[..., 1:-1, :-2]) / dx**2
#     # d2u_dy2 = (u_padded[..., 2:, 1:-1] - 2 * u_padded[..., 1:-1, 1:-1] + u_padded[..., :-2, 1:-1]) / dy**2

#     # lapl = d2u_dx2 + d2u_dy2
#     # return lapl


#     grad_u = grad(u=u,
#                   dx=dx,
#                   dy=dy,
#                   boundary=boundary,
#                   numpy_=numpy_)

#     lapl = divergence(u=grad_u,
#                       dx=dx,
#                       dy=dy,
#                       boundary=boundary,
#                       numpy_=numpy_)

#     return lapl