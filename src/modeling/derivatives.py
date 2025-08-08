import torch

def compute_grads(x, y):
    grads = []

    for i in range(y.shape[1]):
        grad_outs = torch.zeros_like(y)
        grad_outs[:, i, :, :] = 1.0
        
        grad = torch.autograd.grad(y, x, grad_outputs=grad_outs, create_graph=True)[0]

        grads.append(grad)

    return grads

def divergence(v, u, grid_spacing):

    grid_spacing = grid_spacing[0].item()

    grad_x = torch.gradient(v, spacing=grid_spacing, dim=(1, 2), edge_order=2)
    grad_y = torch.gradient(u, spacing=grid_spacing, dim=(1, 2), edge_order=2)

    return v * grad_x[1] + u * grad_x[0] + v * grad_y[1] + u * grad_y[0]