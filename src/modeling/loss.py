import torch
import torch.nn as nn
import torch.nn.functional as F

from derivatives import compute_grads#, divergence

# def laplacian(v, dx, dy):
#     dy = float(dy[0].item())
#     dx = float(dx[0].item())

#     gy, gx = torch.gradient(v, spacing=(dy, dx), dim=(1, 2), edge_order=2)
#     gyy, _  = torch.gradient(gy, spacing=(dy, dx), dim=(1, 2), edge_order=2)
#     _,  gxx = torch.gradient(gx, spacing=(dy, dx), dim=(1, 2), edge_order=2)
#     return gyy + gxx

def laplacian(v, dx, dy):
    v_pad = F.pad(v, (1,1,1,1), mode='replicate')
    lap = (v_pad[:,2:,1:-1] - 2*v + v_pad[:,:-2,1:-1]) / dy**2 + (v_pad[:,1:-1,2:] - 2*v + v_pad[:,1:-1,:-2]) / dx**2
    
    return lap

def divergence(u, v, dx, dy):
    dy = float(dy[0].item())
    dx = float(dx[0].item())
    du_dy, du_dx = torch.gradient(u, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    dv_dy, dv_dx = torch.gradient(v, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    return du_dx + dv_dy

class PoissonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, output, delta_x, delta_y, t, delta_t, rho, mu, coords, boundary_mask, loss_weights, u_star_mean, u_star_std):
        delta_x = delta_x.to(output.device).view(-1, 1, 1)
        delta_y = delta_y.to(output.device).view(-1, 1, 1)
        rho = rho.to(output.device).view(-1, 1, 1)
        delta_t = delta_t.to(output.device).view(-1, 1, 1)
        boundary_mask = boundary_mask.to(output.device)

        laplacian_phi = laplacian(output, delta_x, delta_y)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(output[2].detach())
        # ax[1].imshow(laplacian_phi[2].detach())
        # plt.show()
        # exit(0)

        div_u_star = divergence(x[:, 0], 
                                x[:, 1], 
                                delta_x, 
                                delta_y)
        div_u_star = div_u_star * rho / delta_t

        norm = div_u_star.abs().mean(dim=(1,2), keepdim=True).detach().clamp(min=1e-6)
        loss_pde = ((laplacian_phi - div_u_star) / norm) ** 2
        mask = (boundary_mask - 1).abs()
        loss_pde = torch.mean(loss_pde * mask)
        
        batch, height, width = output.shape

        # base_y, base_x = (x[:, 2] * (height - 1)).long(), (x[:, 3] * (width - 1)).long()
        bc_y, bc_x = x[:, 4].long(), x[:, 5].long()

        base_x, base_y = torch.meshgrid(
                torch.linspace(0, 19, width),
                torch.linspace(0, 19, height),
                indexing="ij"
        )

        base_x = base_x.long()
        base_y = base_y.long()

        bc_y = base_y + bc_y
        bc_x = base_x + bc_x

        B, H, W = output.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, device=output.device),
            torch.arange(W, device=output.device),
            indexing='ij'
        )
        yy = yy.expand(B, -1, -1)
        xx = xx.expand(B, -1, -1)

        off_y = x[:, 4].long()
        off_x = x[:, 5].long()

        bc_y = yy + off_y
        bc_x = xx + off_x

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # fig, ax = plt.subplots(2, 2)
        # ax = ax.flatten()
        # ax[0].imshow((yy + off_y)[10].detach())
        # ax[1].imshow((xx + off_x)[10].detach())
        # ax[2].imshow((yy - off_y)[10].detach())
        # ax[3].imshow((xx - off_x)[10].detach())
        # plt.show()

        valid = (boundary_mask == 1) & ((off_y != 0) | (off_x != 0)) & (bc_y >= 0) & (bc_y < H) & (bc_x >= 0) & (bc_x < W)

        bidx = torch.arange(B, device=output.device)[:, None, None]
        p_center = output[bidx, yy, xx]
        p_interior = output[bidx, bc_y.clamp(0, H-1), bc_x.clamp(0, W-1)]

    #     import matplotlib.pyplot as plt
    #     from matplotlib.patches import Rectangle
    #     import numpy as np
    #     plt.ioff()
    #     plt.clf()
    # # 
    #     x = np.arange(width)
    #     y = np.arange(height)
    #     X, Y = np.meshgrid(x, y)   
    #     # 
    #     U = off_x[0]                      # (n, m)
    #     V = off_y[0]                     # (n, m)
    #     # 
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     for i in range(height):
    #         for j in range(width):
    #             rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
    #                          edgecolor='gray', facecolor='none', linewidth=0.5)
    #             ax.add_patch(rect)
    #     # ax.scatter(X, Y, color='black', s=10, alpha=0.8, label='Grid points')
    #     q = ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, pivot='tail')
    #     ax.set_aspect('equal')

    #     plt.scatter(xx[0], yy[0], color="black")
    #     plt.scatter((xx + off_x)[0], (yy + off_y)[0], color="red")
    #     print(xx+off_x)
    #     plt.show()

    #     exit(0)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # fig, ax = plt.subplots(2, 2)
        # ax = ax.flatten()
        # ax[0].imshow(p_center[10].detach())
        # ax[1].imshow(p_interior[10].detach())
        # # ax[2].imshow((yy - off_y)[10].detach())
        # # ax[3].imshow((xx - off_x)[10].detach())
        # plt.show()

        den = torch.where(off_y != 0, delta_y, delta_x)

        loss_bc_p = torch.mean((((p_center - p_interior) / den) ** 2)[valid])   #.sum() #/ (valid.sum().clamp_min(1))

        loss_bc_p += torch.mean(output[:, 1:-1, -1] ** 2)

        loss_bc_p += torch.mean(torch.relu(-output)**2)

        # valid_mask = (bc_y >= 0) & (bc_y < height) & (bc_x >= 0) & (bc_x < width)

        # batch_idx = torch.arange(batch, device=output.device)[:, None, None].expand(batch, height, width)

        # neighbor_vals = output[batch_idx, bc_y.clamp(0, height - 1), bc_x.clamp(0, width - 1)]
        # center_vals   = output[batch_idx, base_y, base_x]

        # diff = center_vals - neighbor_vals

        # loss_bc_p = ((diff / delta_x) ** 2)[valid_mask].sum() / (valid_mask.sum() + 1e-6)

        # loss_bc_p = torch.mean((diff[valid_mask] / delta_x) ** 2)

        # loss_bc_p += torch.mean(output[:, 1:-1, -1] ** 2)

        loss_bc = loss_bc_p

        loss = loss_weights[0] * loss_pde + loss_weights[1] * loss_bc

        return loss, loss_pde, loss_bc, torch.tensor(0), torch.tensor(0)

class NavierStokesLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, output, delta_x, delta_y, t, delta_t, rho, mu, coords, boundary, loss_weights):
        # Reformat constants
        delta_x = delta_x.to(output.device).view(-1, 1, 1)
        rho = rho.to(output.device).view(-1, 1, 1, 1)
        mu = mu.to(output.device).view(-1, 1, 1, 1)
        delta_t = delta_t.to(output.device).view(-1, 1, 1)

        # Compute derivatives of output w.r.t. input
        grad_uy, grad_ux, grad_p = compute_grads(x, output)

        # Split derivatives for brevity
        duy_dt = grad_uy[:, 0]
        duy_dy = grad_uy[:, 1]
        duy_dx = grad_uy[:, 2]

        dux_dt = grad_ux[:, 0]
        dux_dy = grad_ux[:, 1]
        dux_dx = grad_ux[:, 2]

        dp_dt = grad_p[:, 0]
        dp_dy = grad_p[:, 1]
        dp_dx = grad_p[:, 2]

        # Compute divergence of u
        div_u = dux_dx + duy_dy

        # Reformat derivate of u w.r.t. time
        du_dt = torch.stack([duy_dt, dux_dt], dim=1)

        # Compute u dot del u
        u_dot_del_u_x = x[:, 2] * dux_dx + x[:, 1] * dux_dy
        u_dot_del_u_y = x[:, 2] * duy_dx + x[:, 1] * duy_dy
        u_dot_del_u = torch.stack([u_dot_del_u_y, u_dot_del_u_x], dim=1)

        # Compute the Laplacian of u
        d2uy_dx2 = torch.autograd.grad(duy_dx, x,
                                        grad_outputs=torch.ones_like(duy_dx),
                                        create_graph=True)[0][:, 2, :, :]
        d2uy_dy2 = torch.autograd.grad(duy_dy, x,
                                        grad_outputs=torch.ones_like(duy_dy),
                                        create_graph=True)[0][:, 1, :, :]
        d2ux_dx2 = torch.autograd.grad(dux_dx, x,
                                        grad_outputs=torch.ones_like(dux_dx),
                                        create_graph=True)[0][:, 2, :, :]
        d2ux_dy2 = torch.autograd.grad(dux_dy, x,
                                        grad_outputs=torch.ones_like(dux_dy),
                                        create_graph=True)[0][:, 1, :, :]
        laplacian_uy = d2uy_dx2 + d2uy_dy2
        laplacian_ux = d2ux_dx2 + d2ux_dy2
        laplacian_u = torch.stack([laplacian_uy, laplacian_ux], dim=1)

        # Reformat the gradient of p
        grad_p_ = torch.stack([dp_dy, dp_dx], dim=1)

        # Compute divergence loss
        loss_div = torch.mean(div_u ** 2)

        # Compute momentum loss
        loss_mom = torch.mean((du_dt + u_dot_del_u + (1 / rho) * grad_p_ - mu * laplacian_u) ** 2)

        # Compute boundary condition loss
        loss_bc_uy = torch.mean((output[:, 1, 0, :] - output[:, 1, 1, :]) ** 2) + torch.mean((output[:, 1, -1, :] - output[:, 1, -2, :]) ** 2)
        loss_bc_ux = torch.mean((output[:, 0, 0, :] - output[:, 0, 1, :]) ** 2
                            + torch.mean((output[:, 0, -1, :] - output[:, 0, -2, :]) ** 2)
                            + torch.mean((output[:, 0, 1:-1, 0] - 1) ** 2))
        loss_bc_p = torch.mean((output[:, 2, 1:-1, -1] - output[:, 2, 1:-1, -2]) ** 2)
        loss_bc = loss_bc_uy + loss_bc_ux + loss_bc_p

        # Weighted sum of losses
        loss = loss_weights[0] * loss_div + loss_weights[1] * loss_mom + loss_weights[2] * loss_bc

        return loss, loss_div, loss_mom, loss_bc, torch.tensor(0)

# class PoissonLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input, output, target, delta_x, delta_t, rho, coords, loss_weights):
#         delta_x = delta_x.to(output.device).view(-1, 1, 1)
#         rho = rho.to(output.device).view(-1, 1, 1)
#         delta_t = delta_t.to(output.device).view(-1, 1, 1)

#         laplacian_phi = laplacian(output, coords)
#         div_u_star = divergence(input[:, 0], input[:, 1], delta_x)
#         div_u_star = div_u_star * rho / delta_t

#         loss_pde = torch.mean(((laplacian_phi[:, 1:-1, 1:-1] - div_u_star[:, 1:-1, 1:-1]) ** 2))

#         '''
#         TODO:
#         To future Gregor: I really have no idea what is wrong here. Take a break and then look at it again.
#         Maybe you will find something useful on Git. Also try to visualize thing.

#         Sincerely,
#         Past Gregor
#         '''


#         loss_bc_inlet = torch.mean((output[:, :, 0] - output[:, :, 1]) ** 2)
#         loss_bc_outlet = torch.mean(output[:, :, -1] ** 2)

#         loss_mse = torch.tensor(0)

#         loss_bc = loss_bc_outlet + loss_bc_inlet
        
#         loss_symmetry = torch.mean((output - torch.flip(output, dims=[-2])) ** 2)

#         loss = loss_pde + loss_weights[0] * loss_bc + loss_weights[2] * loss_symmetry #+ loss_weights[1] * loss_mse

#         return loss, loss_pde, loss_weights[0] * loss_bc, loss_weights[1] * loss_mse, loss_weights[2] * loss_symmetry

# class NavierStokesLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input, output, delta_x, delta_t, rho, mu, loss_weights):
#         delta_x = delta_x.to(output.device).view(-1, 1, 1)
#         rho = rho.to(output.device).view(-1, 1, 1, 1)
#         # print(rho.shape)
#         mu = mu.to(output.device).view(-1, 1, 1, 1)
#         delta_t = delta_t.to(output.device).view(-1, 1, 1)

#         grads = []
        
#         for i in range(output.shape[1]):
#             grad_outs = torch.zeros_like(output)
#             grad_outs[:, i, :, :] = 1.0
            
#             # grad = torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True)[0]
#             grad = torch.autograd.grad(output, input, grad_outputs=grad_outs, create_graph=True)[0]

#             grads.append(grad)


#         grad_uy, grad_ux, grad_p = grads

#         duy_dt = grad_uy[:, 0]
#         duy_dy = grad_uy[:, 1]
#         duy_dx = grad_uy[:, 2]

#         dux_dt = grad_ux[:, 0]
#         dux_dy = grad_ux[:, 1]
#         dux_dx = grad_ux[:, 2]

#         dp_dt = grad_p[:, 0]
#         dp_dy = grad_p[:, 1]
#         dp_dx = grad_p[:, 2]

#         div_u = dux_dx + duy_dy

#         du_dt = torch.stack([duy_dt, dux_dt], dim=1)

#         u_dot_del_u_x = input[:, 2] * dux_dx + input[:, 1] * dux_dy
#         u_dot_del_u_y = input[:, 2] * duy_dx + input[:, 1] * duy_dy

#         u_dot_del_u = torch.stack([u_dot_del_u_y, u_dot_del_u_x], dim=1)

#         d2uy_dx2 = torch.autograd.grad(duy_dx, input,
#                                         grad_outputs=torch.ones_like(duy_dx),
#                                         create_graph=True)[0][:, 2, :, :]

#         d2uy_dy2 = torch.autograd.grad(duy_dy, input,
#                                         grad_outputs=torch.ones_like(duy_dy),
#                                         create_graph=True)[0][:, 1, :, :]
        
#         d2ux_dx2 = torch.autograd.grad(dux_dx, input,
#                                         grad_outputs=torch.ones_like(dux_dx),
#                                         create_graph=True)[0][:, 2, :, :]

#         d2ux_dy2 = torch.autograd.grad(dux_dy, input,
#                                         grad_outputs=torch.ones_like(dux_dy),
#                                         create_graph=True)[0][:, 1, :, :]

#         laplacian_uy = d2uy_dx2 + d2uy_dy2
#         laplacian_ux = d2ux_dx2 + d2ux_dy2

#         laplacian_u = torch.stack([laplacian_uy, laplacian_ux], dim=1)

#         grad_p_ = torch.stack([dp_dy, dp_dx], dim=1)

#         loss_div = torch.mean(div_u ** 2)
#         loss_mom = torch.mean((du_dt + u_dot_del_u + (1 / rho) * grad_p_ - mu * laplacian_u) ** 2)

#         loss_bc_uy = torch.mean((output[:, 1, 0, :] - output[:, 1, 1, :]) ** 2) + torch.mean((output[:, 1, -1, :] - output[:, 1, -2, :]) ** 2)
        
#         loss_bc_ux = torch.mean((output[:, 0, 0, :] - output[:, 0, 1, :]) ** 2
#                             + torch.mean((output[:, 0, -1, :] - output[:, 0, -2, :]) ** 2)
#                             + torch.mean((output[:, 0, 1:-1, 0] - 1) ** 2))
        
#         loss_bc_p = torch.mean((output[:, 2, 1:-1, -1] - output[:, 2, 1:-1, -2]) ** 2)

#         loss_bc = loss_bc_uy + loss_bc_ux + loss_bc_p

#         loss = loss_div + loss_mom + loss_bc

#         return loss, loss_div, loss_mom