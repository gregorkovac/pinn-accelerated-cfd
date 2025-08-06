import torch

def compute_grads(x, y):
    grads = []

    for i in range(y.shape[1]):
        grad_outs = torch.zeros_like(y)
        grad_outs[:, i, :, :] = 1.0
        
        grad = torch.autograd.grad(y, x, grad_outputs=grad_outs, create_graph=True)[0]

        grads.append(grad)

    return grads