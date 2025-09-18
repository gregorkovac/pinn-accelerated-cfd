import torch
import torch.nn.functional as F

def compute_grads(x, y):
    grads = []

    for i in range(y.shape[1]):
        grad_outs = torch.zeros_like(y)
        grad_outs[:, i, :, :] = 1.0
        
        grad = torch.autograd.grad(y, x, grad_outputs=grad_outs, create_graph=True)[0]

        grads.append(grad)

    return grads

def laplacian(v, dx, dy):
    v_pad = F.pad(v, (1,1,1,1), mode='replicate')
    lap = (v_pad[:,2:,1:-1] - 2*v + v_pad[:,:-2,1:-1]) / dy**2 + (v_pad[:,1:-1,2:] - 2*v + v_pad[:,1:-1,:-2]) / dx**2
    
    return lap

def laplacian_2d(v, dx, dy):
    dx = dx.view(-1, 1, 1, 1)
    dy = dy.view(-1, 1, 1, 1)

    v_pad = F.pad(v, (1, 1, 1, 1), mode='replicate')

    lap = ((v_pad[:, :, 2:, 1:-1] - 2 * v + v_pad[:, :, :-2, 1:-1]) / dy**2 +
           (v_pad[:, :, 1:-1, 2:] - 2 * v + v_pad[:, :, 1:-1, :-2]) / dx**2)
    
    return lap

def divergence(u, v, dx, dy):
    dy = float(dy[0].item())
    dx = float(dx[0].item())
    du_dy, du_dx = torch.gradient(u, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    dv_dy, dv_dx = torch.gradient(v, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    return du_dx + dv_dy