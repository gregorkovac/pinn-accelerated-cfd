import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from derivatives import divergence

cmap = mpl.cm.Blues(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[10:,:-1])

def plot_full(ax, x, grid_spacing, boundary_image = np.zeros((20, 20)), cbar=None):

    u = x[:2]
    p = x[2]

    # u = np.transpose(u, (1, 2, 0))

    width, height = np.shape(u)[1], np.shape(u)[2]
    x_max = (width - 1) * grid_spacing
    y_max = (height - 1) * grid_spacing

    for a in ax:
        a.clear()
        
    boundary_image = boundary_image.numpy()[::-1, :]

    X, Y = np.meshgrid(np.linspace(0, x_max, width) + 0.5 * grid_spacing, 
                       np.linspace(0, y_max, height) + 0.5 * grid_spacing)
    # ax.contourf(X, Y, p, cmap=cmap)
    im0 = ax[0].imshow(p[::-1, :] * (1 - boundary_image), cmap=cmap, extent=[0, x_max + grid_spacing, 
                                    0, y_max + grid_spacing])
    
    if cbar:
        for cb in cbar:
            cb.remove()
    
    cbar0 = plt.colorbar(im0, ax=ax[0])

    ax[0].quiver(X, Y, u[1,:,:], u[0,:,:], color="white",) #alpha = np.minimum(np.ones(np.shape(boundary_image)), 0.5 + np.abs(boundary_image - 1)))

    # ax[0].imshow(boundary_image, cmap="Greys", extent=[0, x_max + grid_spacing, 
    #                                 0, y_max + grid_spacing], alpha=0.5*boundary_image)
    

    ax[0].set_title(f"p = {np.mean(p):.2f} +- {np.std(p):.2f}")

    ax[1].imshow(boundary_image)

    return [cbar0]
    
def plot_poisson(ax, x, y, boundary, cbar=None, delta_t=1, rho=1, in_=None, dy=1, dx=1):
    for a in ax:
        a.clear()

    # grad_phi = torch.autograd.grad(x, in_, grad_outputs=torch.ones_like(x), create_graph=True)[0][2:4]
    # grad_phi = np.gradient(x.detach().squeeze().numpy(), dy, dx)[0]
    grad_phi = np.gradient(x.detach().squeeze().numpy(), dy, dx, edge_order=2)[0]

    u = in_[:2] - (delta_t / rho) * grad_phi
    u = u.detach().numpy()

    u_star = in_[:2].detach().numpy()

    x = x.detach().squeeze().numpy()
    y = y.detach().squeeze().numpy()

    x_ = x * (1 - boundary.detach().numpy())
    y_ = y * (1 - boundary.detach().numpy())
    u = u * (1 - boundary.detach().numpy())
    u_star = u_star * (1 - boundary.detach().numpy())

    # div_u = divergence(np.expand_dims(u[0], axis=0), np.expand_dims(u[1], axis=0), dx.detach().numpy()*np.ones((1, 20, 20)), dy.detach().numpy()*np.ones((1, 20, 20)), np.expand_dims(boundary.detach().numpy(), axis=0), numpy_=True)
    # div_u = np.squeeze(div_u)

    # div_u = div_u * (1 - boundary.detach().numpy())
    # div_u = div_u ** 2


    grad_ux = np.gradient(u[1], dy, dx, edge_order=2)
    grad_uy = np.gradient(u[0], dy, dx, edge_order=2)
    div_u = grad_ux[1] + grad_uy[0]
    div_u = div_u * (1 - boundary.detach().numpy())

    im0 = ax[0].imshow(x_, cmap=cmap)
    im1 = ax[1].imshow(y_, cmap=cmap)
    # im2 = ax[2].imshow(boundary, cmap=cmap)

    height, width = x_.shape[0], x_.shape[1]
    X, Y = np.meshgrid(np.linspace(0, width, width) + 0.5, 
                       np.linspace(0, height, height) + 0.5)
    
    im2 = ax[2].quiver(X, Y, u[0], u[1], color="black", alpha = 1)
    im2 = ax[2].quiver(X, Y, u_star[0], u_star[1], color="red", alpha = 1)

    im3 = ax[3].imshow(div_u)

    ax[0].set_title(f"p = {np.mean(x_):.2f} +- {np.std(x_):.2f}")
    ax[1].set_title(f"p = {np.mean(np.array(y_)):.2f} +- {np.std(np.array(y_)):.2f}")

    if cbar:
        for cb in cbar:
            cb.remove()

    cbar0 = plt.colorbar(im0, ax=ax[0])
    cbar1 = plt.colorbar(im1, ax=ax[1])
    cbar2 = plt.colorbar(im2, ax=ax[2])
    cbar3 = plt.colorbar(im3, ax=ax[3])

    return [cbar0, cbar1, cbar2, cbar3]