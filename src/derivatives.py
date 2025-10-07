import numpy as np
import torch

def derivative(u, h, boundary, numpy_=False):
    # Compute forward and backward shifted arrays with Torch/NumPy
    if numpy_:
        boundary = boundary.astype(bool)

        u_fwd = np.roll(u, 1, axis=-1)
        b_fwd = np.roll(boundary, 1, axis=-1)

        u_bwd = np.roll(u, -1, axis=-1)
        b_bwd = np.roll(boundary, -1, axis=-1)

        if isinstance(h, (int, float)):
            h_fwd = h
        else:
            h_fwd = np.roll(h, 1, axis=-1)
    else:
        boundary = boundary.bool()
        
        u_fwd = torch.roll(u, 1, dims=-1)
        b_fwd = torch.roll(boundary, 1, dims=-1)

        u_bwd = torch.roll(u, -1, dims=-1)
        b_bwd = torch.roll(boundary, -1, dims=-1)

        if isinstance(h, (int, float)):
            h_fwd = h
        else:
            h_fwd = torch.roll(h, 1, dims=-1)

    # b_bwd = b_bwd & (1 - b_fwd)

    b_fwd[boundary] = 0
    b_bwd[boundary] = 0

    # Compute central difference mask
    b_cent = 1 - (b_bwd | b_fwd)

    # if len(np.shape(u)) == 3:
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(1, 3)
    #     ax[0].imshow(b_fwd[0])
    #     ax[1].imshow(b_bwd[0])
    #     ax[2].imshow(b_cent[0])
    #     plt.show()

    # Set edge cases
    b_fwd[..., 0] = 1
    b_bwd[..., -1] = 1
    b_cent[..., 0] = 0
    b_cent[..., -1] = 0

    # Compute central, forward, and backward difference
    diff_cent = (u_bwd - u_fwd) / (h + h_fwd)
    diff_fwd = (u_bwd - u) / h
    diff_bwd = (u - u_fwd) / h_fwd

    # Compute final difference array
    diff = (b_cent * diff_cent + b_bwd * diff_bwd + b_fwd * diff_fwd)

    return diff

def grad(u, dx, dy, boundary, numpy_=False):
    # Prepare transpose versions of the input
    if numpy_:
        u_t = np.swapaxes(u, -2, -1)
        b_t = np.swapaxes(boundary, -2, -1)
    else:
        u_t = u.transpose(-2, -1)
        b_t = boundary.transpose(-2, -1)

    # Derivative along x axis
    du_dx = derivative(u=u,
                       h=dx,
                       boundary=boundary,
                       numpy_=numpy_)
    
    # Derivative of transpose along x axis
    #  = derivative along y axis
    du_dy = derivative(u=u_t,
                       h=dy,
                       boundary=b_t,
                       numpy_=numpy_)
    
    # Tranpose the derivative back
    if numpy_:
        du_dy = np.swapaxes(du_dy, -2, -1)
    else:
        du_dy = du_dy.transpose(-2, -1)

    if numpy_:
        return np.stack([du_dy, du_dx], axis=0)
    else:
        return torch.stack([du_dy, du_dx], dim=1)

def unpack_grad(grad_u):
    du_dy, du_dx = grad_u

    duy_dy, dux_dy = du_dy
    duy_dx, dux_dx = du_dx

    return duy_dy, dux_dy, duy_dx, dux_dx

def divergence(u, dx, dy, boundary, numpy_=False):
    grad_u = grad(u=u,
                  dx=dx,
                  dy=dy,
                  boundary=boundary,
                  numpy_=numpy_)

    duy_dy, dux_dy, duy_dx, dux_dx = unpack_grad(grad_u)

    div = dux_dx + duy_dy

    return div

def convective(u, dx, dy, boundary, numpy_=False):
    grad_u = grad(u=u,
                  dx=dx,
                  dy=dy,
                  boundary=boundary,
                  numpy_=numpy_)

    du_dy, du_dx = grad_u

    u_y, u_x = u[..., 0, :, :], u[..., 1, :, :]

    conv = u_x * du_dx + u_y * du_dy

    return conv

def laplacian(u, dx, dy, boundary, numpy_=False):
    grad_u = grad(u=u,
                  dx=dx,
                  dy=dy,
                  boundary=boundary,
                  numpy_=numpy_)
    
    du_dy, du_dx = grad_u

    grad_du_dy = grad(u=du_dy,
                   dx=dx,
                   dy=dy,
                   boundary=boundary,
                   numpy_=numpy_)
    
    grad_du_dx = grad(u=du_dx,
                   dx=dx,
                   dy=dy,
                   boundary=boundary,
                   numpy_=numpy_)
    
    d2u_dy2 = grad_du_dy[0]
    d2u_dx2 = grad_du_dx[1]

    lapl = d2u_dy2 + d2u_dx2

    return lapl

if __name__ == "__main__":
    u = [[[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]],
            [[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]]]
    h = 1
    boundary = [[[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
                [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]]
    
    # grad_u = grad(np.array(u), h, h, np.array(boundary), numpy_=True)
    div_u = laplacian(np.array(u), h, h, np.array(boundary), numpy_=True)

    print(div_u)