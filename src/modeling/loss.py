import torch
import torch.nn as nn
import torch.nn.functional as F

from derivatives import compute_grads, divergence

def laplacian(v, grid_spacing):
    grid_spacing = grid_spacing[0].item()
    grad_y, grad_x = torch.gradient(v, spacing=grid_spacing, dim=(1, 2), edge_order=2)
    
    lapl = torch.gradient(grad_y, spacing=grid_spacing, dim=(1, 2), edge_order=2)[0] + torch.gradient(grad_x, spacing=grid_spacing, dim=(1, 2), edge_order=2)[1]
    
    return lapl

class PoissonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, output, delta_x, delta_y, t, delta_t, rho, mu, coords, boundary, loss_weights):
        delta_x = delta_x.to(output.device).view(-1, 1, 1)
        delta_y = delta_y.to(output.device).view(-1, 1, 1)
        rho = rho.to(output.device).view(-1, 1, 1)
        delta_t = delta_t.to(output.device).view(-1, 1, 1)
        boundary = boundary.to(output.device)

        laplacian_phi = laplacian(output, delta_x)

        # grad_phi = torch.autograd.grad(output, x,
        #                         grad_outputs=torch.ones_like(output),
        #                         create_graph=True)[0]

        # dphi_dx = grad_phi[:, 2, :, :]
        # dphi_dy = grad_phi[:, 3, :, :]

        # d2phi_dx2 = torch.autograd.grad(dphi_dx, x,
        #                                  grad_outputs=torch.ones_like(dphi_dx),
        #                                  create_graph=True)[0][:, 2, :, :]

        # d2phi_dy2 = torch.autograd.grad(dphi_dy, x,
        #                                  grad_outputs=torch.ones_like(dphi_dy),
        #                                  create_graph=True)[0][:, 3, :, :]

        # laplacian_phi = d2phi_dx2 + d2phi_dy2

        # print(laplacian_phi.mean())

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # fig, ax = plt.subplots(nrows=2, ncols=2)
        # ax = ax.flatten()
        # ax[0].imshow(output[10].detach().numpy())
        # ax[0].set_title("Phi")
        # ax[1].imshow(dphi_dx[10].detach().numpy())
        # ax[1].set_title("dPhi/dx")
        # ax[2].imshow(dphi_dy[10].detach().numpy())
        # ax[2].set_title("dPhi/dy")
        # ax[3].imshow(laplacian_phi[10].detach().numpy())
        # ax[3].set_title("Laplacian")
        # plt.show()

        div_u_star = divergence(x[:, 0], x[:, 1], delta_x)
        div_u_star = div_u_star * rho / delta_t

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(div_u_star[10].detach().numpy())
        # plt.show()

        # loss_pde_raw = (laplacian_phi - div_u_star) ** 2
        # loss_pde = torch.mean(loss_pde_raw) / (div_u_star.mean().detach() + 1e-8)
        loss_pde = torch.mean(((laplacian_phi - div_u_star) / (div_u_star.abs().mean().detach() + 1e-8)) ** 2)
        # loss_pde = torch.mean(laplacian_phi ** 2)
        # loss_pde = torch.mean(grad_phi ** 2)

        # loss_bc_inlet = torch.mean((output[:, :, 0] - output[:, :, 1]) ** 2)
        # loss_bc_outlet = torch.mean(output[:, :, -1] ** 2)
        # loss_bc = loss_bc_outlet + loss_bc_inlet

        # loss_bc_p = torch.mean((output[:, :, -1] - output[:, :, -2]) ** 2)
        loss_bc_p = torch.mean(output[:, :, -1] ** 2)
        loss_bc_p += torch.mean((output[:, 0, :] - output[:, 1, :]) ** 2)
        loss_bc_p += torch.mean((output[:, -1, :] - output[:, -2, :]) ** 2)
        loss_bc_p += torch.mean((output[:, :, 0] - output[:, :, 1]) ** 2)

        # cylinder_x, cylinder_y, cylinder_r = boundary[:, 0], boundary[:, 1], boundary[:, 2]
        # grid_y = coords[:, 0, :, :]  # shape: (B, H, W)
        # grid_x = coords[:, 1, :, :]  # shape: (B, H, W)

        # # Extract centers and radii: (B, 1, 1) for broadcasting
        # cx = boundary[:, 0].view(-1, 1, 1)
        # cy = boundary[:, 1].view(-1, 1, 1)
        # r  = boundary[:, 2].view(-1, 1, 1)

        # Mask of cylinder interior: (B, H, W)
        # mask = ((grid_x / delta_x - cx) ** 2 + (grid_y / delta_y - cy) ** 2) <= (r ** 2)
        mask = x[:, 4]

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(mask[10].detach().numpy())
        # plt.show()

        # Velocity inside cylinder
        # u_inside = x[:, 0] * mask
        # v_inside = x[:, 1] * mask

        # loss_bc_cylinder = torch.mean(u_inside ** 2) + torch.mean(v_inside ** 2)
        # loss_bc_p += torch.mean((output * mask) ** 2)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(output[10].detach().numpy())
        # plt.colorbar()
        # plt.show()
        # plt.imshow(mask[10].detach().numpy())
        # plt.colorbar()
        # plt.show()
        # plt.imshow(((output * mask) ** 2)[10].detach().numpy())
        # plt.colorbar()
        # plt.show()

        # loss_bc_cylinder = torch.mean((output * mask) ** 2)

        loss_bc = loss_bc_p #+ loss_bc_cylinder

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