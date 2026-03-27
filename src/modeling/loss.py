import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

# from derivatives import compute_grads, divergence, laplacian, laplacian_2d, compute_spatial_derivatives, derivative

# from derivatives import *


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
        # delta_x = x[:, 1:2, 0, 0].view(-1, 1, 1)
        # delta_y = x[:, 2:3, 0, 0].view(-1, 1, 1)

        # boundary_mask = x[:, 1:0]
        delta_x = x[:, 1:2, 0, 0].view(-1, 1, 1)
        delta_y = x[:, 2:3, 0, 0].view(-1, 1, 1)


        # print(float(delta_x[0].item()))
        # print(float(delta_y[0].item()))
        # exit(0)

        # div_u_star = div_u_star.unsqueeze(1)

        # Compute the laplacian of pressure
        laplacian_phi = laplacian(phi, float(delta_x[0].item()), float(delta_y[0].item()), None)
        # laplacian_phi = laplacian(phi, 1, 1, None)

        # loss_pde = torch.abs(laplacian_phi - div_u_star)
        # loss_pde = (laplacian_phi - div_u_star) ** 2

        # loss_pde = loss_pde[boundary_mask != 1]

        # loss_pde = torch.mean(loss_pde)

        # loss_pde = torch.max(torch.abs(laplacian_phi - div_u_star))
        loss_pde = torch.sqrt(torch.mean(((laplacian_phi - div_u_star) ** 2))) #/ (torch.sqrt(torch.mean(div_u_star ** 2)) + 1e-6)

        # loss_pde = loss_pde + 0.1 * torch.max(torch.abs(laplacian_phi - div_u_star))

        # loss_nonneg = torch.mean(torch.relu(-output) ** 2)
        loss_nonneg = torch.tensor(0, device=output.device)

        loss_bc = torch.tensor(0, device=output.device)

        loss_div = torch.tensor(0, device=output.device)

        loss = loss_weights[0] * loss_pde + loss_weights[1] * loss_bc + loss_weights[2] * loss_nonneg + loss_weights[3] * loss_div

        return loss, loss_pde, loss_bc, loss_nonneg, loss_div

init_losses = {}

class NavierStokesLoss(nn.Module):
    def __init__(self, enforceBC):
        super().__init__()
        self.enforceBC = enforceBC

    def forward(self, x, output, loss_weights):
        # Reformat constants
        # delta_x = delta_x.to(output.device).view(-1, 1, 1)
        # delta_y = delta_y.to(output.device).view(-1, 1, 1)
        # rho = rho.to(output.device).view(-1, 1, 1, 1)
        # mu = mu.to(output.device).view(-1, 1, 1, 1)
        rho = x[:, 12:13, 0, 0].view(-1, 1, 1, 1)
        mu = x[:, 13:14, 0, 0].view(-1, 1, 1, 1)
        # delta_t = delta_t.to(output.device).view(-1, 1, 1)
        boundary_mask = x[:, 14]

        delta_x = x[0, 15:16, 0, 0] #.view(-1, 1, 1, 1)
        delta_y = x[0, 16:17, 0, 0] #.view(-1, 1, 1, 1)

        # plt.ioff()
        # fig, ax = plt.subplots(2, 2)
        # ax = ax.flatten()
        # im0 = ax[0].imshow(output[0, 0].detach().cpu().numpy())
        # ax[0].set_title("u[0]")
        # plt.colorbar(im0)
        # im1 = ax[1].imshow(output[0, 1].detach().cpu().numpy())
        # ax[1].set_title("u[1]")
        # plt.colorbar(im1)


        # output = self.enforceBC(x, output)

        # im2 = ax[2].imshow(output[0, 0].detach().cpu().numpy())
        # ax[2].set_title("u_bc[0]")
        # plt.colorbar(im2)
        # im3 = ax[3].imshow(output[0, 1].detach().cpu().numpy())
        # ax[3].set_title("u_bc[1]")
        # plt.colorbar(im3)
        # plt.show()

        ux = output[:, 0]
        uy = output[:, 1]
        p = output[:, 2]

        delta_x = float(delta_x.item())
        delta_y = float(delta_y.item())

        # boundary_mask *= 0
        boundary_mask_u = boundary_mask.unsqueeze(1).repeat(1, 2, 1, 1)

        # boundary_mask = 1 - boundary_mask

        # grad_ux = grad(u = ux,
        #                 dx=delta_x,
        #                 dy=delta_y,
        #                 boundary=boundary_mask * 0)
        
        # grad_uy= grad(u = uy,
        #                 dx=delta_x,
        #                 dy=delta_y,
        #                 boundary=boundary_mask * 0)
                        
        grad_p = grad(u = p,
                        dx=delta_x,
                        dy=delta_y,
                        boundary=boundary_mask * 0)
        
        # plt.ioff()
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(ux[0].cpu().detach().numpy())
        # ax[1].imshow(grad_ux[0, 0].cpu().detach().numpy())
        # ax[2].imshow(grad_ux[0, 1].cpu().detach().numpy())
        # plt.show()

        # dux_dy = grad_ux[..., 0]
        # dux_dx = grad_ux[..., 1]
        
        # duy_dy = grad_uy[..., 0]
        # duy_dx = grad_uy[..., 1]

        dp_dx = grad_p[..., 0]
        dp_dy = grad_p[..., 1]


        div_u = divergence(u=torch.stack([ux, uy], dim=1),
                           dx=delta_x,
                           dy=delta_y,
                           boundary=boundary_mask_u * 0)
        
        u_dot_del_u = convective(u = torch.stack([ux, uy], axis=1),
                           dx = delta_x,
                           dy = delta_y,
                           boundary = boundary_mask_u * 0)
        
 
        grad_p = torch.stack([dp_dx, dp_dy], dim=1)

        laplacian_u = laplacian(u = torch.stack([ux, uy], axis=1),
                           dx = delta_x,
                           dy = delta_y,
                           boundary = boundary_mask_u * 0)
        
        # div_u_dot_del_u = divergence(u = u_dot_del_u,
        #                              dx = delta_x,
        #                              dy = delta_y,
        #                              boundary=boundary_mask_u * 0)
        
        # laplacian_p = laplacian(u = p,
        #                    dx = delta_x,
        #                    dy = delta_y,
        #                    boundary = boundary_mask_u * 0)
        # plt.ioff()
        # fig, ax = plt.subplots(3, 3)
        # ax = ax.flatten()
        # ax[0].imshow(div_u[0].detach().cpu().numpy())
        # ax[1].imshow(u_dot_del_u[0, 0].detach().cpu().numpy())
        # ax[2].imshow(u_dot_del_u[0, 1].detach().cpu().numpy())
        # ax[3].imshow(grad_p[0, 0].detach().cpu().numpy())
        # ax[4].imshow(grad_p[0, 1].detach().cpu().numpy())
        # ax[5].imshow(laplacian_u[0, 0].detach().cpu().numpy())
        # ax[6].imshow(laplacian_u[0, 1].detach().cpu().numpy())

        # plt.show()

        # TODO: PRESSURE BCS!

        loss_div = div_u ** 2
        # loss_div = torch.abs(div_u)

        # loss_div[boundary_mask != 0] = 0

        loss_div = loss_div * (1 - boundary_mask)
        loss_div = torch.mean(loss_div)

        res_mom = (u_dot_del_u + (1 / rho) * grad_p - mu * laplacian_u) ** 2
        # res_mom = (u_dot_del_u + grad_p - mu * laplacian_u) ** 2
        res_mom = res_mom * (1 - boundary_mask_u)

        # loss_mom = torch.mean(res_mom[:, 0]) + torch.mean(res_mom[:, 1])
        # loss_mom = torch.mean(res_mom[:, 0]) #** (1 / 2)
        # loss_nonneg_p = torch.mean(res_mom[:, 1]) #** (1 / 2)

        # res_mom[boundary_mask_u != 0] = 0

        loss_mom = torch.mean(res_mom)

        # mask_dy = 1 - x[:, 6]
        # mask_dx = 1 - x[:, 7]
        # loss_bc = torch.mean((mask_dy * dux_dy) ** 2 + (mask_dy * duy_dy) ** 2 + (mask_dx * dux_dx) ** 2 + (mask_dx * duy_dx) ** 2)

        # mask_dy = 1 - x[:, 10]
        # mask_dx = 1 - x[:, 11]
        # loss_bc += torch.mean((mask_dy * dp_dy) ** 2 + (mask_dx * dp_dx) ** 2)

        # # loss_nonneg_p = torch.mean(torch.relu(-output[:, 2])**2) * 0

        # loss_bc += torch.sum((uy[..., 0] - uy[..., -1]) ** 2) + torch.sum((ux[..., 0] - ux[..., -1]) ** 2)

        # loss_nonneg_p = (div_u_dot_del_u + (1 / rho) * laplacian_p) ** 2
        # loss_nonneg_p = torch.mean(loss_nonneg_p)

        loss_nonneg_p = torch.tensor(0)
        loss_bc = torch.tensor(0)

        # if epoch < 50:
        #     dynamic_weight_mom = 0
        # else:
        #     dynamic_weight_mom = min(1, (epoch - 50) / 20)

        # Weighted sum of losses
        loss = loss_weights[0] * loss_div + loss_weights[1] * loss_mom + loss_weights[2] * loss_bc + loss_weights[3] * loss_nonneg_p

        return loss, loss_div, loss_mom, loss_bc, loss_nonneg_p