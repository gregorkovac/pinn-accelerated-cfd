import torch
import torch.nn as nn
import torch.nn.functional as F

from derivatives import compute_grads#, divergence

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

def neumann_bc_loss(off_y, off_x, boundary_mask, output, delta_y, delta_x):


    # Extract output shape
    if len(output.shape) == 3:
        batch, height, width = output.shape
    else:
        batch, channels, height, width = output.shape

    # Create a coordinate grid
    yy, xx = torch.meshgrid(
        torch.arange(height, device=output.device),
        torch.arange(width, device=output.device),
        indexing='ij'
    )
    yy = yy.expand(batch, -1, -1)
    xx = xx.expand(batch, -1, -1)

    # # Extract pressure Neumann offsets
    # off_y = x[:, 4].long()
    # off_x = x[:, 5].long()

    # Compute offset coordinates
    bc_y = yy + off_y
    bc_x = xx + off_x

    # Check if it is valid to apply each Neumann BC
    valid = (boundary_mask == 1) & ((off_y != 0) | (off_x != 0)) & (bc_y >= 0) & (bc_y < height) & (bc_x >= 0) & (bc_x < width)

    # Batch indices
    bidx = torch.arange(batch, device=output.device)[:, None, None]

    if len(output.shape) == 3:
        # Base pressure values
        p_base = output[bidx, yy, xx]

        # Offset pressure values
        # p_offset = output[bidx, bc_y.clamp(0, height-1), bc_x.clamp(0, width-1)]
        p_offset = output[bidx, bc_y, bc_x]
    else:
        # Base pressure values
        p_base = output[bidx, :, yy, xx]

        # Offset pressure values
        # p_offset = output[bidx, :, bc_y.clamp(0, height-1), bc_x.clamp(0, width-1)]
        p_offset = output[bidx, :, bc_y, bc_x]

    # Dynamic spacing for BC in x and y direction
    spacing = torch.where(off_y != 0, delta_y, delta_x)

    if len(output.shape) > 3:
        spacing = spacing.unsqueeze(-1).repeat(1, 1, 1, channels)
        valid = valid.unsqueeze(-1).repeat(1, 1, 1, channels)

    # Compute BC loss
    loss_bc_p = torch.mean((((p_base - p_offset) / spacing) ** 2)[valid])

    if torch.isnan(loss_bc_p):
        return torch.tensor(0, dtype=torch.float32)

    return loss_bc_p

class PoissonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, output, delta_x, delta_y, t, delta_t, rho, mu, coords, boundary_mask, loss_weights):
        delta_x = delta_x.to(output.device).view(-1, 1, 1)
        delta_y = delta_y.to(output.device).view(-1, 1, 1)
        rho = rho.to(output.device).view(-1, 1, 1)
        delta_t = delta_t.to(output.device).view(-1, 1, 1)
        boundary_mask = boundary_mask.to(output.device)

        # Compute the laplacian of pressure
        laplacian_phi = laplacian(output, delta_x, delta_y)

        # Compute the divergence of the approximate velocity
        div_u_star = divergence(x[:, 0], 
                                x[:, 1], 
                                delta_x, 
                                delta_y)
        
        # Scale
        div_u_star = div_u_star * rho / delta_t

        # Compute the mean of divergence of u star - for normalizing the loss
        norm = div_u_star.abs().mean(dim=(1,2), keepdim=True).detach().clamp(min=1e-6)

        # Compute the PDE loss
        loss_pde = ((laplacian_phi - div_u_star) / norm) ** 2

        # Mask out boundary values and compute the mean
        mask = (boundary_mask - 1).abs()
        loss_pde = torch.mean(loss_pde * mask)
        
        # Extract output dimensions
        # batch, height, width = output.shape

        # # Extract the pressure Neumann boundary condition offsets
        # bc_y, bc_x = x[:, 4].long(), x[:, 5].long()

        # # Create a coordinate grid
        # base_x, base_y = torch.meshgrid(
        #         torch.linspace(0, width - 1, width),
        #         torch.linspace(0, height - 1, height),
        #         indexing="ij"
        # )
        # base_x = base_x.long()
        # base_y = base_y.long()

        # # Compute offset indices
        # bc_y = base_y + bc_y
        # bc_x = base_x + bc_x

        loss_bc_p = neumann_bc_loss(off_y=x[:, 4].long(),
                                    off_x=x[:, 5].long(),
                                    boundary_mask=boundary_mask,
                                    output=output,
                                    delta_y=delta_y,
                                    delta_x=delta_x)

        # Add pressure gauge on outflow
        p_dirichlet_mask = x[:, -1]
        loss_bc_p += torch.mean(output[p_dirichlet_mask == 1] ** 2)

        loss_bc = loss_bc_p

        # Loss to enforce non-negative values
        loss_nonneg = torch.mean(torch.relu(-output)**2)

        loss = loss_weights[0] * loss_pde + loss_weights[1] * loss_bc + loss_weights[2] * loss_nonneg

        return loss, loss_pde, loss_bc, torch.tensor(0), torch.tensor(0)

init_losses = {}

class NavierStokesLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, output, delta_x, delta_y, t, delta_t, rho, mu, coords, boundary_mask, loss_weights):
        # Reformat constants
        delta_x = delta_x.to(output.device).view(-1, 1, 1)
        delta_y = delta_y.to(output.device).view(-1, 1, 1)
        rho = rho.to(output.device).view(-1, 1, 1, 1)
        mu = mu.to(output.device).view(-1, 1, 1, 1)
        delta_t = delta_t.to(output.device).view(-1, 1, 1)

        # u_dirichlet = x[:, 3:5]
        # u_dirichlet_mask = x[:, 5].unsqueeze(1).repeat(1, 2, 1, 1)

        # u_bc = output[:, :2] * (1 - u_dirichlet_mask) + u_dirichlet * u_dirichlet_mask

        # output_bc = torch.cat([u_bc, output[:, 2:3]], dim=1)

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

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(duy_dt[0].detach())
        # ax[1].imshow(duy_dy[0].detach())
        # ax[2].imshow(duy_dx[0].detach())
        # plt.show()

        du_dt = torch.stack([duy_dt, dux_dt], dim=1)

        div_u = divergence(output[:, 0], output[:, 1], delta_x, delta_y)

        # print(grad_first_channel)
        # print(grad_first_channel.shape)
        # exit(0)


        u_dot_del_u_x = output[:, 1] * dux_dx + output[:, 0] * dux_dy # ux * dux/dx + uy * dux/dy 
        u_dot_del_u_y = output[:, 1] * duy_dx + output[:, 0] * duy_dy # ux * duy/dx + uy * duy/dy
        u_dot_del_u = torch.stack([u_dot_del_u_y, u_dot_del_u_x], dim=1)

        grad_p_ = torch.stack([dp_dy, dp_dx], dim=1)

        laplacian_u = laplacian_2d(output[:, :2, :, :], delta_x, delta_y)

        mask = (boundary_mask - 1).abs()
        # mask[:, 1:-1, 0] = 1
        # mask[:, 1:-1, -1] = 1
        # print(mask.shape)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(mask[0].detach())
        # plt.show()


        # Compute divergence loss
        loss_div = div_u ** 2
        
        loss_div = torch.mean(loss_div * mask)
        

        # Compute momentum loss
        loss_mom = (du_dt + u_dot_del_u + (1 / rho) * grad_p_ - mu * laplacian_u) ** 2
        # loss_mom = torch.mean(loss_mom * mask.unsqueeze(1).repeat(1, 2, 1, 1))
        loss_mom = torch.mean(loss_mom)
        
        loss_bc_p = neumann_bc_loss(off_y=x[:, 10].long(),
                                    off_x=x[:, 11].long(),
                                    boundary_mask=boundary_mask,
                                    output=output[:, 2],
                                    delta_y=delta_y,
                                    delta_x=delta_x)
    

        # Add pressure gauge on outflow
        p_dirichlet_mask = x[:, 9]
        loss_bc_p_dirichlet = torch.mean(output[:, 2][p_dirichlet_mask == 1] ** 2)

        if not torch.isnan(loss_bc_p_dirichlet):
            loss_bc_p += loss_bc_p_dirichlet

        loss_bc_u = neumann_bc_loss(off_y=x[:, 6].long(),
                                    off_x=x[:, 7].long(),
                                    boundary_mask=boundary_mask,
                                    output=output[:, :2],
                                    delta_y=delta_y,
                                    delta_x=delta_x)
        
        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(x[0, 7].detach())
        # plt.colorbar()
        # plt.show()

        u_dirichlet_mask = x[:, 5].unsqueeze(-1).repeat(1, 1, 1, 2)
        u_dirichlet_val = x[:, 3:5]

        # Extract the first two channels from output
        u_output = output[:, :2]

        # Apply mask correctly (select masked elements for both tensors)
        u_output_masked = u_output.permute(0, 2, 3, 1)[u_dirichlet_mask == 1]
        u_val_masked = u_dirichlet_val.permute(0, 2, 3, 1)[u_dirichlet_mask == 1]
        
        # weights = torch.ones_like(u_dirichlet_mask)

        # weights = (u_dirichlet_val == 0).float()
        # weights = weights.all(dim=1).unsqueeze(-1)
        # weights = weights * 99 + 1

        # # print(u_dirichlet_val.permute(0, 2, 3, 1).shape)
        # # print(weights.shape)

        # weights = weights.repeat(1, 1, 1, 2)[u_dirichlet_mask == 1]
        # print(weights)
        # exit(0)

        # print(weights.shape)
        # print(u_dirichlet_mask.shape)
        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(weights[0, :, :, 0].detach())
        # plt.show()

        weights = torch.ones_like(u_val_masked)
        weights[u_val_masked == 0] = 1.0
        loss_bc_u_dirichlet = (u_output_masked - u_val_masked) ** 2

        loss_bc_u_dirichlet = torch.mean(weights * loss_bc_u_dirichlet)

        # sdf = x[:, -1]

        # k = 10.0
        # alpha = torch.exp(-k * sdf)

        # # import matplotlib.pyplot as plt
        # # plt.ioff()
        # # plt.imshow(alpha[0].detach())
        # # plt.show()

        # # Apply to u and v boundary conditions
        # loss_bc_u_dirichlet = torch.mean(alpha * (u_output - u_dirichlet_val) ** 2)

        if not torch.isnan(loss_bc_u_dirichlet):
            loss_bc_u += loss_bc_u_dirichlet

        loss_bc = loss_bc_p + loss_bc_u

        loss_nonneg_p = torch.mean(torch.relu(-output[:, 2])**2)
        # loss_nonneg_p = torch.tensor(0)

        # loss_nonzero = output.abs()
        # mask_nonzero = loss_nonzero < 0.0001
        # loss_nonzero = loss_nonzero * mask_nonzero * mask
        # loss_nonzero = loss_nonzero.mean() * 100

        # print(loss_nonzero)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(loss_nonzero[0, 1].detach())
        # plt.show()

        global init_losses
        if len(init_losses) == 0:
            init_losses["loss_div"] = loss_div.detach().item()
            init_losses["loss_mom"] = loss_mom.detach().item()
            init_losses["loss_bc"] = loss_bc.detach().item()
            init_losses["loss_nonneg_p"] = loss_nonneg_p.detach().item()

        if init_losses != None:
            loss_div /= (init_losses["loss_div"] + 1e-10)
            loss_mom /= (init_losses["loss_mom"] + 1e-10)
            loss_bc /= (init_losses["loss_bc"] + 1e-10)
            loss_nonneg_p /= (init_losses["loss_nonneg_p"] + 1e-10)


        # TODO: Google mass flux
        inflow_flux = (output[:, 1, 1:-1, 0]).sum()
        outflow_flux = (output[:, 1, 1:-1, -1]).sum()
        loss_mass_flux = ((inflow_flux - outflow_flux) / outflow_flux) ** 2

        # print(f"[{inflow_flux}]   [{outflow_flux}]   -> {loss_mass_flux}")

        # Weighted sum of losses
        loss = loss_weights[0] * loss_div + loss_weights[1] * loss_mom + loss_weights[2] * loss_bc + loss_weights[3] * loss_nonneg_p + loss_mass_flux

        return loss, loss_div, loss_mom, loss_bc, loss_mass_flux

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