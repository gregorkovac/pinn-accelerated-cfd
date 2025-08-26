import torch
import torch.nn as nn
import torch.nn.functional as F

from derivatives import compute_grads#, divergence

def laplacian(v, dx, dy):
    # v: (B, H, W)
    dy = float(dy[0].item())
    dx = float(dx[0].item())
    # grads along (y, x) because v is (B, H, W) with dim=(1, 2)
    gy, gx = torch.gradient(v, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    gyy, _  = torch.gradient(gy, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    _,  gxx = torch.gradient(gx, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    return gyy + gxx

def divergence(u, v, dx, dy):
    # u,v: (B, H, W) velocities on cell centers
    dy = float(dy[0].item())
    dx = float(dx[0].item())
    du_dy, du_dx = torch.gradient(u, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    dv_dy, dv_dx = torch.gradient(v, spacing=(dy, dx), dim=(1, 2), edge_order=2)
    return du_dx + dv_dy

# def laplacian(v, grid_spacing):
#     grid_spacing = grid_spacing[0].item()
#     grad_y, grad_x = torch.gradient(v, spacing=grid_spacing, dim=(1, 2), edge_order=2)
    
#     lapl = torch.gradient(grad_y, spacing=grid_spacing, dim=(1, 2), edge_order=2)[0] + torch.gradient(grad_x, spacing=grid_spacing, dim=(1, 2), edge_order=2)[1]
    
#     return lapl

# def laplacian(v, dx, dy):
#     dx = float(dx[0].item())
#     dy = float(dy[0].item())
#     # grads along (y, x) because v is (B, H, W) with dim=(1,2)
#     gy, gx = torch.gradient(v, spacing=(dy, dx), dim=(1, 2), edge_order=2)
#     gyy, _ = torch.gradient(gy, spacing=(dy, dx), dim=(1, 2), edge_order=2)
#     _, gxx = torch.gradient(gx, spacing=(dy, dx), dim=(1, 2), edge_order=2)
#     return gyy + gxx

class PoissonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, output, delta_x, delta_y, t, delta_t, rho, mu, coords, boundary_mask, loss_weights):
        delta_x = delta_x.to(output.device).view(-1, 1, 1)
        delta_y = delta_y.to(output.device).view(-1, 1, 1)
        rho = rho.to(output.device).view(-1, 1, 1)
        delta_t = delta_t.to(output.device).view(-1, 1, 1)
        boundary_mask = boundary_mask.to(output.device)

        laplacian_phi = laplacian(output, delta_x, delta_y)

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

        div_u_star = divergence(x[:, 0], x[:, 1], delta_x, delta_y)
        div_u_star = div_u_star * rho / delta_t

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(div_u_star[10].detach().numpy())
        # plt.show()

        # loss_pde_raw = (laplacian_phi - div_u_star) ** 2
        # loss_pde = torch.mean(loss_pde_raw) / (div_u_star.mean().detach() + 1e-8)
        # loss_pde = torch.mean(((laplacian_phi - div_u_star) / (div_u_star.abs().mean().detach() + 1e-8)) ** 2)
        # loss_pde = torch.mean(laplacian_phi ** 2)
        # loss_pde = torch.mean(grad_phi ** 2)

        # loss_pde = ((laplacian_phi - div_u_star) / (div_u_star.abs().mean().detach() + 1e-8)) ** 2
        # mask = (x[:, 4] - 1).abs()
        # loss_pde = torch.mean(loss_pde * mask)

        loss_pde = ((laplacian_phi - div_u_star) / (div_u_star.abs().mean().detach() + 1e-8)) ** 2
        # mask = (x[:, 4] - 1).abs()
        # mask = (torch.norm(x[:, 4:6], dim=1) - 1).abs()
        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(x[0, 4].detach())
        # plt.show()
        # plt.imshow(mask[0].detach())
        # plt.show()
        # exit(0)
        # mask[:, :, 0] = 1
        # mask[:, :, -1] = 1
        # mask[:, 0, :] = 1
        # mask[:, -1, :] = 1
        # mask[:, :8, 8:10] = 1


        # mask = x[:, 4] * 0
        # mask[:, 0, :] = 1
        # mask[:, -1, :] = 1
        # mask[:, :8, 8:10] = 1
        # mask[:, :, 0] = 1
        # mask[:, :, -1] = 1

        
        # mask = (mask - 1).abs()

        mask = (boundary_mask - 1).abs()
        # import matplotlib.pyplot as plt
        # plt.ioff()
        # # plt.imshow(boundary_mask[0].detach())
        # # plt.show()
        # plt.imshow(mask[0].detach())
        # plt.show()


        loss_pde = torch.mean(loss_pde * mask)

        # rhs = div_u_star
        # rhs = rhs - rhs.mean(dim=(1,2), keepdim=True)
        # loss_pde = torch.mean(((laplacian_phi[:, 1:-1, 1:-1] - rhs[:, 1:-1, 1:-1]) /
        #                (rhs[:, 1:-1, 1:-1].abs().mean().detach() + 1e-8)) ** 2)

        # loss_bc_inlet = torch.mean((output[:, :, 0] - output[:, :, 1]) ** 2)
        # loss_bc_outlet = torch.mean(output[:, :, -1] ** 2)
        # loss_bc = loss_bc_outlet + loss_bc_inlet

        # loss_bc_p = torch.mean((output[:, :, -1] - output[:, :, -2]) ** 2)
        # loss_bc_p = torch.mean(output[:, :, -1] ** 2)
        # loss_bc_p += torch.mean((output[:, 0, :] - output[:, 1, :]) ** 2)
        # loss_bc_p += torch.mean((output[:, -1, :] - output[:, -2, :]) ** 2)
        # loss_bc_p += torch.mean((output[:, :, 0] - output[:, :, 1]) ** 2)
        # loss_bc_p = torch.mean((output[:, -1, :] - output[:, -2, :]) ** 2)

        # loss_bc_p = torch.mean((output[:, 0, :] - output[:, 1, :]) ** 2)
        # loss_bc_p += torch.mean((output[:, 1:-1, 0] - output[:, 1:-1, 1]) ** 2)
        # loss_bc_p += torch.mean((output[:, 1:-1, -1] - output[:, 1:-1, -2]) ** 2)

        # loss_bc_p_bottom = (output[:, 0, :] - output[:, 1, :]).pow(2).mean()
        # loss_bc_p_left = (output[:, 1:-1, 0] - output[:, 1:-1, 1]).pow(2).mean()
        # loss_bc_p_right = (output[:, 1:-1, -1] - output[:, 1:-1, -2]).pow(2).mean()

        # loss_bc_p = loss_bc_p_bottom + loss_bc_p_left + loss_bc_p_right

        # loss_bc_p += torch.mean(output ** 2)

        # loss_bc_p = torch.mean(output[:, -1, -1] ** 2)

        # loss_bc_p = torch.mean(output[:, 0, :] ** 2)
        # loss_bc_p += torch.mean(output[:, 1:-1, 0] ** 2)
        # loss_bc_p += torch.mean(output[:, 1:-1, -1] ** 2)

        # loss_bc_p = torch.mean((output[:, -1] - output[:, -2]) ** 2)

        # loss_bc_p += torch.mean(output[:, -1, 10] ** 2)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # y = output[0].detach().numpy() * 0
        # y[0, :] = 100
        # plt.imshow(y)
        # plt.show()
        # exit(0)

        # output[:, 0, :] = 10000

        # loss_bc_p = torch.tensor(0)
        # loss_bc_p = torch.mean((output[:, -1, :] - output[:, -2, :]) ** 2)
        # loss_bc_p = torch.mean((output[:, 1:-1, -1] - output[:, 1:-1, -2]) ** 2)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(x[0, 0].detach())
        # plt.show()
        # exit(0)

        # dpdy, dpdx = torch.gradient(output, spacing=(float(delta_y[0].item()), float(delta_x[0].item())), dim=(1, 2), edge_order=2)

        # loss_bc_p = torch.mean(output[:, :, -1] ** 2)

        # loss_bc_p = torch.mean(dpdy[:, 0] ** 2)
        # loss_bc_p += torch.mean(dpdy[:, -1] ** 2)
        # loss_bc_p += torch.mean(dpdx[:, :, 0] ** 2)
        # loss_bc_p += torch.mean(dpdx[:, :8, 8] ** 2)
        # loss_bc_p += torch.mean(dpdx[:, :8, 9] ** 2)
        # loss_bc_p += torch.mean(dpdy[:, 8, 8] ** 2)
        # loss_bc_p += torch.mean(dpdy[:, 8, 9] ** 2)

        # Outflow
        # loss_bc_p = torch.mean(output[:, :, -1] ** 2)

        # # # Lower wall
        # # loss_bc_p += torch.mean((output[:, 0, :8] - output[:, 1, :8]) ** 2) + torch.mean((output[:, 0, 10:] - output[:, 1, 10:]) ** 2)
        # loss_bc_p = torch.mean(((output[:, 0, :] - output[:, 1, :]) / (2 * delta_y)) ** 2)

        # # # Upper wall
        # loss_bc_p += torch.mean(((output[:, -1, :] - output[:, -2, :]) / (2 * delta_y)) ** 2)

        # # # Inflow
        # loss_bc_p += torch.mean(((output[:, :, 0] - output[:, :, 1]) / (2 * delta_x)) ** 2)

        # # # # Dam - left side
        # loss_bc_p += torch.mean(((output[:, :8, 8] - output[:, :8, 7]) / (2 * delta_x)) ** 2)

        # # # # Dam - right side
        # loss_bc_p += torch.mean(((output[:, :8, 9] - output[:, :8, 10]) / (2 * delta_x)) ** 2)

        # # # # Dam - top side
        # loss_bc_p += torch.mean(((output[:, 8, 8:10] - output[:, 7, 8:10]) / (2 * delta_y)) ** 2)

        # neumann_idx = (x[:, 2:4] + x[:, 4:6]).long()
        # # neumann_phi = output[neumann_idx[:, 0], neumann_idx[:, 1]]

        base_y, base_x = x[:, 2].long(), x[:, 3].long()
        bc_y, bc_x = x[:, 4].long(), x[:, 5].long()

        # bc_y = base_y + bc_y
        # bc_x = base_x + bc_x

        bc_y = (base_y + bc_y).clamp(0, 19)
        bc_x = (base_x + bc_x).clamp(0, 19)


        loss_bc_p = torch.mean(((output[:, base_y, base_x] - output[:, bc_y, bc_x]) / delta_x) ** 2)

        # loss_bc_p += torch.mean(output[:, 1:-1, -1] ** 2)

        # B, Ny, Nx = output.shape
        # # pressure = output[:, 0]

        # boundary_idx = x[:, 2:4].long()
        # neumann_idx = (x[:, 2:4] + x[:, 4:6]).long()

        # # boundary and neighbor indices
        # iy, ix = boundary_idx[:, 0], boundary_idx[:, 1]
        # jy, jx = neumann_idx[:, 0], neumann_idx[:, 1]

        # # Batch indexing
        # batch_idx = torch.arange(B, device=output.device).view(B, 1, 1)

        # # Get pressure values at boundary and neighbor
        # u_boundary = output[batch_idx, iy, ix]
        # neumann_phi = output[batch_idx, jy, jx]

        # # import matplotlib.pyplot as plt
        # # plt.ioff()
        # # fig, ax = plt.subplots(nrows=1, ncols=3)
        # # ax = ax.flatten()
        # # ax[0].imshow(output[10].detach())
        # # ax[1].imshow(u_boundary[10].detach())
        # # ax[2].imshow(neumann_phi[10].detach())
        # # plt.show()

        # # Compute direction: x[:, 4:6] is direction (dx_dir, dy_dir)
        # direction = x[:, 4:6]  # shape [B, 2, H, W]
        # dx_dir = direction[:, 1]  # [B, H, W]
        # dy_dir = direction[:, 0]  # [B, H, W]

        # # Compute spacing based on direction
        # spacing = torch.where(dx_dir != 0, 2 * delta_x, 2 * delta_y)  # [B, H, W]

        # # Neumann loss with proper delta per direction
        # loss_bc_p = torch.mean(((u_boundary - neumann_phi) / spacing) ** 2)

        # output_flat = output.view(output.shape[0], -1)

        # # # Convert 2D indices (i,j) to flat indices
        # # flat_idx = neumann_idx[:, 0] * Ny + neumann_idx[:, 1]  # shape [Nb]

        # # # Gather values for boundary neighbor points
        # # neumann_phi = output_flat[:, flat_idx]

        # # # print(neumann_phi.shape)
        # # # print(output.shape)
        # # loss_bc_p = torch.mean((neumann_phi - output) ** 2)

        # boundary_idx = x[:, 2:4].long()
        # neumann_idx = (x[:, 2:4] + x[:, 4:6]).long()

        # # print(boundary_idx.shape)
        # # print(neumann_idx.shape)
        
        # flat_boundary = boundary_idx[:, 0] * output.shape[2] + boundary_idx[:, 1]
        # flat_neighbor = neumann_idx[:, 0] * output.shape[2] + neumann_idx[:, 1]

        # flat_boundary = flat_boundary.view(output.shape[0], -1)
        # flat_neighbor = flat_neighbor.view(output.shape[0], -1) 

        # # print(flat_boundary.shape)
        # # print(flat_neighbor.shape)
        
        # # u_boundary = output_flat[:, flat_boundary]
        # # neumann_phi = output_flat[:, flat_neighbor]
        
        # u_boundary = torch.gather(output_flat, 1, flat_boundary)
        # neumann_phi = torch.gather(output_flat, 1, flat_neighbor)

        # # print(u_boundary.shape)
        # # print(neumann_phi.shape)

        # loss_bc_p = torch.mean(((u_boundary - neumann_phi) / delta_x)** 2)

        # exit(0)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(u_boundary[0].detach())
        # plt.show()


        # Bottom wall
        # loss_bc_p = torch.mean(((output[:, 0, 1:-1] - output[:, 1, 1:-1]) / (2 * delta_y)) ** 2)

        # # Left wall
        # loss_bc_p += torch.mean(((output[:, 1:-1, 0] - output[:, 1:-1, 1]) / (2 * delta_x)) ** 2)

        # # Right wall
        # loss_bc_p += torch.mean(((output[:, :-1, -1] - output[:, :-1, -2]) / (2 * delta_x)) ** 2)

        # loss_bc_p = torch.tensor(0)

        # Inflow
        # loss_bc_p += torch.mean(((output[:, -1, 1:-1] - output[:, -2, 1:-1]) / (2 * delta_y)) ** 2)

        # Outflow
        # loss_bc_p = torch.mean(output[:, -1, -1] ** 2)

        # loss_bc_p += torch.mean(output[:, 1, 1] ** 2)

        # cylinder_x, cylinder_y, cylinder_r = boundary[:, 0], boundary[:, 1], boundary[:, 2]
        # grid_y = coords[:, 0, :, :]  # shape: (B, H, W)
        # grid_x = coords[:, 1, :, :]  # shape: (B, H, W)

        # # Extract centers and radii: (B, 1, 1) for broadcasting
        # cx = boundary[:, 0].view(-1, 1, 1)
        # cy = boundary[:, 1].view(-1, 1, 1)
        # r  = boundary[:, 2].view(-1, 1, 1)

        # Mask of cylinder interior: (B, H, W)
        # mask = ((grid_x / delta_x - cx) ** 2 + (grid_y / delta_y - cy) ** 2) <= (r ** 2)
        # mask = x[:, 4]

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