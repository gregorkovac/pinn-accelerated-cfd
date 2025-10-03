import torch
import torch.nn as nn

from derivatives import compute_grads, divergence, laplacian, laplacian_2d, compute_spatial_derivatives, derivative

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
        p_offset = output[bidx, bc_y.clamp(0, height-1), bc_x.clamp(0, width-1)]
        # p_offset = output[bidx, bc_y, bc_x]
    else:
        # Base pressure values
        p_base = output[bidx, :, yy, xx]

        # Offset pressure values
        p_offset = output[bidx, :, bc_y.clamp(0, height-1), bc_x.clamp(0, width-1)]
        # p_offset = output[bidx, :, bc_y, bc_x]

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

        # delta_x = delta_x.expand(-1, 20, 20)
        # delta_y = delta_y.expand(-1, 20, 20)

        # Compute the laplacian of pressure
        laplacian_phi = laplacian(output, delta_x, delta_y, boundary_mask)


        # Compute the divergence of the approximate velocity
        div_u_star = divergence(x[:, 0], 
                                x[:, 1], 
                                delta_x, 
                                delta_y,
                                boundary_mask)

        div_u_star = div_u_star * rho / delta_t

        # laplacian_phi = (1 - boundary_mask) * laplacian_phi
        # div_u_star = (1 - boundary_mask) * div_u_star

        # Scale

        # Compute the mean of divergence of u star - for normalizing the loss
        # norm = div_u_star.abs().mean(dim=(1,2), keepdim=True).detach().clamp(min=1e-6)
        # norm = div_u_star.abs().mean().detach().clamp(min=1e-6)

        # Compute the PDE loss
        # loss_pde = torch.mean((laplacian_phi - div_u_star) ** 2)
        norm = torch.mean(torch.abs(div_u_star)) + 1e-10
        loss_pde = (1 - boundary_mask) * (laplacian_phi - div_u_star) ** 2 
        # mask = (boundary_mask - 1).abs()
        # loss_pde = ((laplacian_phi - div_u_star) / norm) ** 2
        # loss_pde = torch.sum((laplacian_phi - div_u_star) ** 2) / torch.sum(div_u_star ** 2)

        # print(loss_pde.shape)

        loss_pde = torch.mean(loss_pde)

        # Mask out boundary values and compute the mean
        # loss_pde = torch.mean(loss_pde * mask)
        # loss_pde = loss_pde * mask
        
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
        
        # grad_phi = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0][:, 2:4]

        # grad_phi_y, grad_phi_x = torch.gradient(output, spacing=(float(delta_y[0].item()), float(delta_x[0].item())), dim=(1, 2), edge_order=2)
        # grad_phi_x = grad_phi[:, 0]
        # grad_phi_y = grad_phi[:, 1]

        # grad_phi_x = derivative(output, delta_x, boundary_mask)
        # grad_phi_y = derivative(output.transpose(1, 2), delta_y, boundary_mask.transpose(1, 2)).transpose(1, 2)

        # u = x[:, :2] - (delta_t / rho).unsqueeze(1) * grad_phi
        # ux = x[:, 0] - (delta_t / rho) * grad_phi_x
        # uy = x[:, 1] - (delta_t / rho) * grad_phi_y

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(x[5, 0].detach())
        # plt.show()

        # print(x[:, 0].shape)
        # print(grad_phi[:, 0].shape)
        # print((delta_t / rho).unsqueeze(1).shape)

        # div_u = divergence(ux,
        #                    uy,
        #                    delta_x,
        #                    delta_y,
        #                    boundary_mask)

        # div_u = div_u * (1 - boundary_mask)
        
        # loss_div = torch.mean(div_u ** 2)
        loss_div = torch.tensor(0)
        # loss_div = torch.mean(output ** 2)

        # print(loss_div)

        # Add pressure gauge on outflow
        # p_dirichlet_mask = x[:, -1]
        # loss_bc_p += torch.mean(output[p_dirichlet_mask == 1] ** 2)

        loss_nonneg = torch.mean(torch.relu(-output)**2)

        loss_bc = loss_bc_p
        
        loss = loss_weights[0] * loss_pde + loss_weights[1] * loss_bc + loss_weights[2] * loss_nonneg + loss_weights[3] * loss_div

        return loss, loss_pde, loss_bc, loss_nonneg, loss_div

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

        uy = output[:, 0]
        ux = output[:, 1]
        p = output[:, 2]

        # print(len(compute_grads(x, output)))
        # exit(0)

        # grad_uy, grad_ux, grad_p = compute_grads(x, output)

        # # Split derivatives for brevity
        # # duy_dt = grad_uy[:, 0]
        # duy_dy = grad_uy[:, 0]
        # duy_dx = grad_uy[:, 1]

        # # dux_dt = grad_ux[:, 0]
        # dux_dy = grad_ux[:, 0]
        # dux_dx = grad_ux[:, 1]

        # # dp_dt = grad_p[:, 0]
        # dp_dy = grad_p[:, 0]
        # dp_dx = grad_p[:, 1]

        duy_dy, duy_dx = compute_spatial_derivatives(uy, delta_x, delta_y)
        dux_dy, dux_dx = compute_spatial_derivatives(ux, delta_x, delta_y)
        dp_dy, dp_dx = compute_spatial_derivatives(p, delta_x, delta_y)

        # du_dt = torch.stack([duy_dt, dux_dt], dim=1)

        div_u = divergence(u = ux,
                           v = uy,
                           dx = delta_x,
                           dy = delta_y)

        # u_dot_del_u_x = output[:, 1] * dux_dx + output[:, 0] * dux_dy
        # u_dot_del_u_y = output[:, 1] * duy_dx + output[:, 0] * duy_dy
        # u_dot_del_u = torch.stack([u_dot_del_u_y, u_dot_del_u_x], dim=1)

        u_dot_del_u_x = (ux * dux_dx + uy * dux_dy).unsqueeze(1)
        u_dot_del_u_y = (ux * duy_dx + uy * duy_dy).unsqueeze(1)
        u_dot_del_u = torch.cat([u_dot_del_u_y, u_dot_del_u_x], dim=1)
 
        grad_p_ = torch.stack([dp_dy, dp_dx], dim=1)

        # laplacian_u = laplacian_2d(output[:, :2, :, :], delta_x, delta_y)
        laplacian_u = laplacian_2d(torch.cat([uy.unsqueeze(1), ux.unsqueeze(1)], dim=1), delta_x, delta_y)

        loss_div = torch.mean(div_u ** 2)

        # loss_mom = torch.mean((du_dt + u_dot_del_u + (1 / rho) * grad_p_ - mu * laplacian_u) ** 2)
        loss_mom = torch.mean((u_dot_del_u + (1 / rho) * grad_p_ - mu * laplacian_u) ** 2)

        loss_bc_p = neumann_bc_loss(off_y=x[:, 10].long(),
                                    off_x=x[:, 11].long(),
                                    boundary_mask=boundary_mask,
                                    output=output[:, 2],
                                    delta_y=delta_y,
                                    delta_x=delta_x)

        loss_bc_u = neumann_bc_loss(off_y=x[:, 6].long(),
                                    off_x=x[:, 7].long(),
                                    boundary_mask=boundary_mask,
                                    output=output[:, :2],
                                    delta_y=delta_y,
                                    delta_x=delta_x)
        
    
        loss_bc = loss_bc_p + loss_bc_u

        loss_nonneg_p = torch.mean(torch.relu(-output[:, 2])**2)
        # loss_nonneg_p = torch.mean(output[:, 2] ** 2)

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

        # loss_div = loss_div / (loss_div.detach().mean() + 1e-10)
        # loss_mom = loss_mom / (loss_div.detach().mean() + 1e-10)
        # loss_bc = loss_bc / (loss_div.detach().mean() + 1e-10)
        # loss_nonneg_p = loss_nonneg_p / (loss_div.detach().mean() + 1e-10)

        # Weighted sum of losses
        loss = loss_weights[0] * loss_div + loss_weights[1] * loss_mom + loss_weights[2] * loss_bc + loss_weights[3] * loss_nonneg_p

        return loss, loss_div, loss_mom, loss_bc, loss_nonneg_p