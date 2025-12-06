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

class PoissonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, output, loss_weights):

        # delta_x = delta_x.to(output.device).view(-1, 1, 1)
        # delta_y = delta_y.to(output.device).view(-1, 1, 1)

        phi = output[:, 0:1]
        # div_u_star_orig = x[:, 0]
        boundary_mask = x[:, 5:6]
        u_star = x[:, 6:8]
        p_neumann = x[:, 8:10]
        p_dirichlet = x[:, 3:4]
        p_dirichlet_mask = x[:, 4:5]
        rho = x[:, 10:11]
        delta_t = x[:, 11:12]
        delta_x = x[:, 12:13, 0, 0].view(-1, 1, 1)
        delta_y = x[:, 13:14, 0, 0].view(-1, 1, 1)

        # eroded_mask = -F.max_pool2d(-boundary_mask, kernel_size=3, stride=1, padding=3 // 2)

        # eroded_mask = (eroded_mask > 0.5).float()
        # boundary_mask = eroded_mask

        # plt.ioff()
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(boundary_mask[0].cpu().detach().numpy())
        # ax[1].imshow(eroded_mask[0].cpu().detach().numpy())
        # plt.show()

        # u_star_np = u_star.cpu().detach().numpy()
        # u_star_np = u_star_np.transpose(0, 2, 3, 1)
        # u_star_np = u_star_np[0]
        # div_u_star_orig = divergence(u_star_np,
        #                         float(delta_x[0].item()),
        #                         float(delta_y[0].item()),
        #                         boundary_mask,
        #                         numpy_=True)

        div_u_star = divergence(u_star,
                                float(delta_x[0].item()),
                                float(delta_y[0].item()),
                                boundary_mask)
        
        # plt.ioff()
        # fig, ax = plt.subplots(1, 2)
        # im0 = ax[0].imshow(div_u_star_orig)
        # im1 = ax[1].imshow(div_u_star[0].cpu().detach().numpy())
        # ax[0].set_title("numpy")
        # ax[1].set_title("torch")
        # plt.colorbar(im0)
        # plt.colorbar(im1)
        # plt.show()

        div_u_star = div_u_star.unsqueeze(1)

        div_u_star = div_u_star * rho / delta_t     

        # div_u_star_orig = div_u_star_orig.unsqueeze(1)

        # Compute the laplacian of pressure
        laplacian_phi = laplacian(phi, float(delta_x[0].item()), float(delta_y[0].item()), boundary_mask)

        # plt.ioff()
        # plt.imshow(div_u_star[0, 0].cpu().detach().numpy())
        # plt.show()

        # div_u_star *= (1 - boundary_mask)

        loss_pde = (laplacian_phi - div_u_star) ** 2 #/ ((div_u_star ** 2).mean() + 1e-8)

        # loss_pde = loss_pde * (1 - p_dirichlet_mask)

        # plt.ioff()
        # plt.imshow((boundary_mask[0].float() - p_dirichlet_mask[0, 0].float()).cpu().detach().numpy())
        # plt.show()

        # loss_pde /= ((div_u_star ** 2).mean().detach() + 1e-6)

        # boundary_mask[:, :, 1:-1, 0] = 0
        # boundary_mask[:, :, 1:-1, -1] = 0

        # loss_pde *= (1 - boundary_mask)
        # loss_pde *= (1 - (boundary_mask[0].float() - p_dirichlet_mask[0, 0].float()))
        # loss_pde[loss_pde == 0] = 1e-8

        loss_pde = torch.mean(loss_pde)

        # loss_pde = torch.mean(loss_pde)
        # loss_pde = torch.sqrt(torch.mean(loss_pde))
        # loss_pde = torch.mean(torch.log(loss_pde))


        # loss_nonneg = torch.mean(torch.relu(-output) ** 2)
        loss_nonneg = torch.tensor(0, device=output.device)

        loss_bc = torch.tensor(0, device=output.device)
        # loss_bc = torch.mean(loss_pde_b)

        # grad_phi = grad(phi,
        #                 float(delta_x[0].item()),
        #                 float(delta_y[0].item()),
        #                 boundary_mask)

        # loss_bc = torch.mean((grad_phi[:, 0] * p_neumann[:, 1]) ** 2) + torch.mean((grad_phi[:, 1] * p_neumann[:, 0]) ** 2)

        # loss_bc = torch.mean((phi[p_dirichlet_mask] - p_dirichlet[p_dirichlet_mask]) ** 2)

        loss_div = torch.tensor(0, device=output.device)

        loss = loss_weights[0] * loss_pde + loss_weights[1] * loss_bc + loss_weights[2] * loss_nonneg + loss_weights[3] * loss_div

        return loss, loss_pde, loss_bc, loss_nonneg, loss_div

init_losses = {}

class NavierStokesLoss(nn.Module):
    def __init__(self):
        super().__init__()

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

        uy = output[:, 0]
        ux = output[:, 1]
        p = output[:, 2]

        delta_x = float(delta_x.item())
        delta_y = float(delta_y.item())

        # boundary_mask *= 0
        boundary_mask_u = boundary_mask.unsqueeze(1).repeat(1, 2, 1, 1)

        # boundary_mask = 1 - boundary_mask

        grad_ux = grad(u = ux,
                        dx=delta_x,
                        dy=delta_y,
                        boundary=boundary_mask * 0)
        
        grad_uy= grad(u = uy,
                        dx=delta_x,
                        dy=delta_y,
                        boundary=boundary_mask * 0)
                        
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

        dux_dy = grad_ux[:, 0]
        dux_dx = grad_ux[:, 1]
        
        duy_dy = grad_uy[:, 0]
        duy_dx = grad_uy[:, 1]

        dp_dy = grad_p[:, 0]
        dp_dx = grad_p[:, 1]

        div_u = divergence(u=torch.stack([uy, ux], dim=1),
                           dx=delta_x,
                           dy=delta_y,
                           boundary=boundary_mask_u * 0)

        u_dot_del_u = convective(u = torch.stack([uy, ux], axis=1),
                           dx = delta_x,
                           dy = delta_y,
                           boundary = boundary_mask_u * 0)
 
        grad_p = torch.stack([dp_dy, dp_dx], dim=1)

        laplacian_u = laplacian(u = torch.stack([uy, ux], axis=1),
                           dx = delta_x,
                           dy = delta_y,
                           boundary = boundary_mask_u * 0)

        loss_div = div_u ** 2
        # loss_div = loss_div * (1 - boundary_mask)
        loss_div = torch.mean(loss_div)

        res_mom = (u_dot_del_u + (1 / rho) * grad_p - mu * laplacian_u) ** 2
        # res_mom = res_mom * (1 - boundary_mask_u)
        # loss_mom = torch.mean(loss_mom[:, 0]) + torch.mean(loss_mom[:, 1])
        loss_mom = torch.mean(res_mom[:, 0]) #** (1 / 2)
        loss_nonneg_p = torch.mean(res_mom[:, 1]) #** (1 / 2)
        # loss_mom = torch.mean(loss_mom)

        mask_dy = 1 - x[:, 6]
        mask_dx = 1 - x[:, 7]
        loss_bc = torch.mean((mask_dy * dux_dy) ** 2 + (mask_dy * duy_dy) ** 2 + (mask_dx * dux_dx) ** 2 + (mask_dx * duy_dx) ** 2)

        mask_dy = 1 - x[:, 10]
        mask_dx = 1 - x[:, 11]
        loss_bc += torch.mean((mask_dy * dp_dy) ** 2 + (mask_dx * dp_dx) ** 2)

        # loss_nonneg_p = torch.mean(torch.relu(-output[:, 2])**2) * 0

        loss_bc += torch.sum((uy[..., 0] - uy[..., -1]) ** 2) + torch.sum((ux[..., 0] - ux[..., -1]) ** 2)

        # if epoch < 50:
        #     dynamic_weight_mom = 0
        # else:
        #     dynamic_weight_mom = min(1, (epoch - 50) / 20)

        # Weighted sum of losses
        loss = loss_weights[0] * loss_div + loss_weights[1] * loss_mom + loss_weights[2] * loss_bc + loss_weights[3] * loss_nonneg_p

        return loss, loss_div, loss_mom, loss_bc, loss_nonneg_p