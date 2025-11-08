import torch
import torch.nn.functional as F
import numpy as np

def compute_grads(x, y):
    grads = []

    for i in range(y.shape[1]):
        grad_outs = torch.zeros_like(y)
        grad_outs[:, i, :, :] = 1.0
        
        grad = torch.autograd.grad(y, x, grad_outputs=grad_outs, create_graph=False)[0]

        grads.append(grad)

    return grads

# def compute_spatial_derivatives(u, dx, dy):
#     dx = float(dx[0].item())
#     dy = float(dy[0].item())

#     du_dy, du_dx = torch.gradient(u, spacing=(dy, dx), dim=(1, 2), edge_order=2)

#     return du_dy, du_dx

def compute_spatial_derivatives(u, dx, dy):
    dx = float(dx[0].item())
    dy = float(dy[0].item())
    du_dy, du_dx = torch.gradient(u, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    d2u_dy2, _ = torch.gradient(du_dy, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    _, d2u_dx2 = torch.gradient(du_dx, spacing=(dy, dx), dim=(1, 2), edge_order=2)

    return du_dy, du_dx, d2u_dy2, d2u_dx2

def laplacian(v, dx, dy, boundary):
    # v_pad = F.pad(v, (1,1,1,1), mode='constant', value=0)
    v_pad = F.pad(v, (1,1,1,1), mode='replicate')
    lap = (v_pad[:,2:,1:-1] - 2*v + v_pad[:,:-2,1:-1]) / dy**2 +(v_pad[:,1:-1,2:] - 2*v + v_pad[:,1:-1,:-2]) / dx**2
    
    return lap

# def laplacian(v, dx, dy, boundary, numpy_=False):

#     # dy = float(dy[0].item())
#     # dx = float(dx[0].item())

#     dv_dx = derivative(v, dx, boundary, numpy_)
#     d2v_dx2 = derivative(dv_dx, dx, boundary, numpy_)

#     dv_dy = derivative(v.transpose(1, 2), dy, boundary.transpose(1, 2), numpy_)
#     d2v_dy2 = derivative(dv_dy, dy, boundary.transpose(1, 2), numpy_)
#     d2v_dy2 = d2v_dy2.transpose(1, 2)

#     return d2v_dx2 + d2v_dy2

# def laplacian(v, dx, dy, boundary, numpy_=False):
#     d2v_dx2 = second_derivative(v, dx, boundary, numpy_)
#     d2v_dy2 = second_derivative(v.transpose(1, 2), dy, boundary.transpose(1, 2), numpy_)
#     d2v_dy2 = d2v_dy2.transpose(1, 2)

#     return d2v_dx2 + d2v_dy2

# def laplacian(u, ins, dx, dy, boundary):
#     u = u.clone().requires_grad_(True)

#     d2u_dx2 = torch.autograd.grad(
#         torch.autograd.grad(u, ins, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 2],
#         ins, grad_outputs=torch.ones_like(u[:, 0]), create_graph=True
#     )[0][:, 0].view_as(u) / dx**2

#     d2u_dy2 = torch.autograd.grad(
#         torch.autograd.grad(u, ins, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 2],
#         ins, grad_outputs=torch.ones_like(u[:, 1]), create_graph=True
#     )[0][:, 3].view_as(u) / dy**2

#     return d2u_dx2 + d2u_dy2

def laplacian_2d(v, dx, dy):
    dx = dx.view(-1, 1, 1, 1)
    dy = dy.view(-1, 1, 1, 1)

    # v_pad = F.pad(v, (1, 1, 1, 1), mode='constant', value=0)
    v_pad = F.pad(v, (1, 1, 1, 1), mode='replicate')

    lap = ((v_pad[:, :, 2:, 1:-1] - 2 * v + v_pad[:, :, :-2, 1:-1]) / dy**2 +
           (v_pad[:, :, 1:-1, 2:] - 2 * v + v_pad[:, :, 1:-1, :-2]) / dx**2)
    
    return lap

def divergence(u, v, dx, dy, boundary, numpy_=False):
    dy = float(dy[0].item())
    dx = float(dx[0].item())
    
    du_dy, du_dx = torch.gradient(u, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    dv_dy, dv_dx = torch.gradient(v, spacing=(dy, dx), dim=(1, 2), edge_order=2)

    # du_dx = derivative(u, dx, boundary, numpy_)

    # if numpy_:
    #     dv_dy = derivative(v.transpose(0, 2, 1), dy, boundary.transpose(0, 2, 1), numpy_).transpose(0, 2, 1)
    # else:
    #     dv_dy = derivative(v.transpose(1, 2), dy, boundary.transpose(1, 2), numpy_).transpose(1, 2)

    return du_dx + dv_dy

# def derivative(x, h, boundary, numpy_=False):
#     if numpy_:
#         diff = np.zeros_like(x)
#     else:
#         diff = torch.empty_like(x)

#     # Central differences for inner points
#     central_diff = (x[:, :, 2:] - x[:, :, :-2]) / (2 * h[:, :, 1:-1])
#     # Forward/backward differences for shifted points
#     forward_diff = (x[:, :, 2:] - x[:, :, 1:-1]) / h[:, :, 1:-1]
#     backward_diff = (x[:, :, 1:-1] - x[:, :, :-2]) / h[:, :, 1:-1]

#     # Boundary masks
#     if numpy_:
#         forward_mask = np.roll(boundary, 1, axis=2).astype(bool)[:, :, 1:-1]
#         backward_mask = np.roll(boundary, -1, axis=2).astype(bool)[:, :, 1:-1]
#     else:
#         forward_mask = torch.roll(boundary, 1, dims=2).bool()[:, :, 1:-1]
#         backward_mask = torch.roll(boundary, -1, dims=2).bool()[:, :, 1:-1]

#     central_mask = ~(forward_mask | backward_mask)

#     diff[:, :, 1:-1] = forward_mask * forward_diff + backward_mask * backward_diff + central_mask * central_diff

#     # Use one-sided difference at the edges
#     diff[:, :, 0] = (x[:, :, 1] - x[:, :, 0]) / h[:, :, 0]
#     diff[:, :, -1] = (x[:, :, -1] - x[:, :, -2]) / h[:, :, -1]

#     diff *= (1 - boundary)

#     return diff


# def second_derivative(x, h, boundary, numpy_=False):
#     if numpy_:
#         diff = np.zeros_like(x)
#     else:
#         diff = torch.empty_like(x)

#     # Central differences for inner points
#     central_diff = (x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]) / (h[:, :, 1:-1] ** 2)
#     # Forward/backward differences for points near boundaries
#     forward_diff = (2 * x[:, :, 0:-3] - 5 * x[:, :, 1:-2] + 4 * x[:, :, 2:-1] - x[:, :, 3:]) / (h[:, :, 1:-2] ** 2)
#     backward_diff = (2 * x[:, :, 3:] - 5 * x[:, :, 2:-1] + 4 * x[:, :, 1:-2] - x[:, :, 0:-3]) / (h[:, :, 1:-2] ** 2)

#     # Boundary masks
#     if numpy_:
#         forward_mask = np.roll(boundary, 1, axis=2).astype(bool)[:, :, 1:-2]
#         backward_mask = np.roll(boundary, -1, axis=2).astype(bool)[:, :, 1:-2]
#     else:
#         forward_mask = torch.roll(boundary, 1, dims=2).bool()[:, :, 1:-2]
#         backward_mask = torch.roll(boundary, -1, dims=2).bool()[:, :, 1:-2]

#     central_mask = ~(forward_mask | backward_mask)

#     # Apply masks
#     diff[:, :, 1:-2] = forward_mask * forward_diff + backward_mask * backward_diff + central_mask * central_diff[:, :, :-1]

#     # One-sided second derivatives at edges
#     diff[:, :, 0] = (2 * x[:, :, 0] - 5 * x[:, :, 1] + 4 * x[:, :, 2] - x[:, :, 3]) / (h[:, :, 0] ** 2)
#     diff[:, :, -1] = (2 * x[:, :, -1] - 5 * x[:, :, -2] + 4 * x[:, :, -3] - x[:, :, -4]) / (h[:, :, -1] ** 2)

#     diff *= (1 - boundary)

#     return diff

def derivative(x, h, boundary, numpy_=False):
    deriv = torch.zeros_like(x)
    forward = torch.roll(x, -1, dims=2)
    backward = torch.roll(x, 1, dims=2)
    
    deriv = (forward - backward) / (2 * h)

    deriv *= boundary

    deriv = torch.where(torch.roll(boundary.bool(), -1, dims=2), (x - backward) / h, deriv)
    deriv = torch.where(torch.roll(boundary.bool(), 1, dims=2), (forward - x) / h, deriv)
    
    return deriv

def second_derivative(x, h, boundary, numpy_=False):
    deriv = torch.zeros_like(x)
    forward = torch.roll(x, -1, dims=2)
    backward = torch.roll(x, 1, dims=2)

    deriv = (forward - 2 * x + backward) / (h ** 2)
    deriv *= boundary

    # forward / backward second-order differences near boundaries
    deriv = torch.where(
        torch.roll(boundary.bool(), -1, dims=2),       # next cell is boundary
        (x - 2 * backward + torch.roll(x, 2, dims=2)) / (h ** 2),
        deriv
    )
    deriv = torch.where(
        torch.roll(boundary.bool(), 1, dims=2),        # prev cell is boundary
        (x - 2 * forward + torch.roll(x, -2, dims=2)) / (h ** 2),
        deriv
    )

    return deriv


# def second_derivative(x, h, boundary, numpy_=False):
#     deriv = torch.zeros_like(x)
#     forward = torch.roll(x, -1, dims=2)
#     backward = torch.roll(x, 1, dims=2)
    
#     deriv = (forward - 2 * x + backward) / (h ** 2)

#     deriv *= boundary

#     neighbor_mask = torch.roll(boundary, -1, dims=2) * torch.roll(boundary, 1, dims=2)
#     deriv *= neighbor_mask
    
#     return deriv


# def derivative(x, h, boundary, numpy_=False):
#     if numpy_:
#         diff = np.zeros_like(x)
#     else:
#         diff = torch.empty_like(x)

#     central_diff = (x[:, :, 2:] - x[:, :, :-2]) / (2 * h[:, :, 1:-1])
#     forward_diff = (x[:, :, 2:] - x[:, :, 1:-1]) / h[:, :, 1:-1]
#     backward_diff = (x[:, :, 1:-1] - x[:, :, :-2]) / h[:, :, 1:-1]
    
#     if numpy_:
#         forward_diff_mask = np.roll(boundary, 1, axis=2).astype(bool)[:, :, 1:-1]
#         backward_diff_mask = np.roll(boundary, -1, axis=2).astype(bool)[:, :, 1:-1]
#     else:
#         forward_diff_mask = torch.roll(boundary, 1, dims=2).bool()[:, :, 1:-1]
#         backward_diff_mask = torch.roll(boundary, -1, dims=2).bool()[:, :, 1:-1]

#     central_diff_mask = ~(forward_diff_mask | backward_diff_mask)

#     diff[:, :, 1:-1] = forward_diff_mask * forward_diff + backward_diff_mask * backward_diff + central_diff_mask * central_diff
#     diff[:, :, 0] = (x[:, :, 1] - x[:, :, 0]) / h[:, :, 0]
#     diff[:, :, -1] = (x[:, :, -1] - x[:, :, -2]) / h[:, :, -1]

#     return diff

# def second_derivative(x, h, boundary, numpy_=False):
#     if numpy_:
#         diff = np.zeros_like(x)
#     else:
#         diff = torch.empty_like(x)

#     central_diff = (x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]) / (h[:, :, 1:-1] ** 2)
#     forward_diff = (2 * x[:, :, 0:-3] - 5 * x[:, :, 1:-2] + 4 * x[:, :, 2:-1] - x[:, :, 3:]) / (h[:, :, 1:-2] ** 2)
#     backward_diff = (2 * x[:, :, 3:] - 5 * x[:, :, 2:-1] + 4 * x[:, :, 1:-2] - x[:, :, 0:-3]) / (h[:, :, 1:-2] ** 2)
    
#     if numpy_:
#         forward_diff_mask = np.roll(boundary, 1, axis=2).astype(bool)[:, :, 1:-2]
#         backward_diff_mask = np.roll(boundary, -1, axis=2).astype(bool)[:, :, 1:-2]
#     else:
#         forward_diff_mask = torch.roll(boundary, 1, dims=2).bool()[:, :, 1:-2]
#         backward_diff_mask = torch.roll(boundary, -1, dims=2).bool()[:, :, 1:-2]

#     central_diff_mask = ~(forward_diff_mask | backward_diff_mask)

#     diff[:, :, 1:-2] = forward_diff_mask * forward_diff + backward_diff_mask * backward_diff + central_diff_mask * central_diff[:, :, :-1]
#     diff[:, :, 0] = (2*x[:, :, 0] - 5*x[:, :, 1] + 4*x[:, :, 2] - x[:, :, 3]) / (h[:, :, 0] ** 2)
#     diff[:, :, -1] = (2*x[:, :, -1] - 5*x[:, :, -2] + 4*x[:, :, -3] - x[:, :, -4]) / (h[:, :, -1] ** 2)

#     return diff