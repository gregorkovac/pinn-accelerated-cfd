import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, "..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from derivatives import *
from numeric_method.numeric_method import compute_discrete_laplacian

class PoissonLoss(nn.Module):
    def __init__(self, enforceBC):
        super().__init__()
        self.enforceBC = enforceBC

    def forward(self, x, output, loss_weights):
        phi = output[:, 0:1]
        div_u_star = x[:, 0:1]
        delta_x = x[:, 1:2, 0, 0].view(-1, 1, 1)
        delta_y = x[:, 2:3, 0, 0].view(-1, 1, 1)

        # Compute the laplacian of pressure
        laplacian_phi = laplacian(phi, float(delta_x[0].item()), float(delta_y[0].item()), None)

        # PDE loss
        loss_pde = torch.sqrt(torch.mean(((laplacian_phi - div_u_star) ** 2)))

        # Other loss terms; currently unused
        loss_nonneg = torch.tensor(0, device=output.device)
        loss_bc = torch.tensor(0, device=output.device)
        loss_div = torch.tensor(0, device=output.device)

        # Weighted sum of losses
        loss = loss_weights[0] * loss_pde + loss_weights[1] * loss_bc + loss_weights[2] * loss_nonneg + loss_weights[3] * loss_div

        return loss, loss_pde, loss_bc, loss_nonneg, loss_div

class NavierStokesLoss(nn.Module):
    def __init__(self, enforceBC):
        super().__init__()
        self.enforceBC = enforceBC

    def forward(self, x, output, loss_weights):
        rho = x[:, 12:13, 0, 0].view(-1, 1, 1, 1)
        mu = x[:, 13:14, 0, 0].view(-1, 1, 1, 1)
        boundary_mask = x[:, 14]
        delta_x = x[0, 15:16, 0, 0]
        delta_y = x[0, 16:17, 0, 0]
        ux = output[:, 0]
        uy = output[:, 1]
        p = output[:, 2]

        delta_x = float(delta_x.item())
        delta_y = float(delta_y.item())

        boundary_mask_u = boundary_mask.unsqueeze(1).repeat(1, 2, 1, 1)

        # Compute gradient of pressure
        grad_p = grad(u = p,
                        dx=delta_x,
                        dy=delta_y,
                        boundary=boundary_mask * 0)

        # Extract pressure gradient terms
        dp_dx = grad_p[..., 0]
        dp_dy = grad_p[..., 1]
    
        # Stack the pressure gradient
        grad_p = torch.stack([dp_dx, dp_dy], dim=1)

        # Compute the divergence of velocity
        div_u = divergence(u=torch.stack([ux, uy], dim=1),
                           dx=delta_x,
                           dy=delta_y,
                           boundary=boundary_mask_u * 0)

        # Compute the convective term
        u_dot_del_u = convective(u = torch.stack([ux, uy], axis=1),
                           dx = delta_x,
                           dy = delta_y,
                           boundary = boundary_mask_u * 0)
        
        # Compute the laplacian of velocity
        laplacian_u = laplacian(u = torch.stack([ux, uy], axis=1),
                           dx = delta_x,
                           dy = delta_y,
                           boundary = boundary_mask_u * 0)

        # Compute the divergence residual
        loss_div = div_u ** 2

        # Use only loss outside the boundary
        loss_div = loss_div * (1 - boundary_mask)

        # Compute divergence loss
        loss_div = torch.mean(loss_div)

        # Compute the momentum residual
        res_mom = (u_dot_del_u + (1 / rho) * grad_p - mu * laplacian_u) ** 2

        # Keep the momentum residual only inside the boundary
        res_mom = res_mom * (1 - boundary_mask_u)

        # Compute the momentum loss
        loss_mom = torch.mean(res_mom)

        # Other loss terms; currently unused
        loss_nonneg_p = torch.tensor(0)
        loss_bc = torch.tensor(0)

        # Weighted sum of losses
        loss = loss_weights[0] * loss_div + loss_weights[1] * loss_mom + loss_weights[2] * loss_bc + loss_weights[3] * loss_nonneg_p

        return loss, loss_div, loss_mom, loss_bc, loss_nonneg_p