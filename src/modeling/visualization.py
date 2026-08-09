import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
# from derivatives import divergence
import sys
import os
sys.path.insert(0, "..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from derivatives import *

cmap = "viridis"

def plot_full(ax, x, grid_spacing, boundary_image = np.zeros((20, 20)), cbar=None):

    u = x[:2]
    p = x[2]

    width, height = np.shape(u)[1], np.shape(u)[2]
    x_max = (width - 1) * grid_spacing
    y_max = (height - 1) * grid_spacing

    for a in ax:
        a.clear()
        
    X, Y = np.meshgrid(np.linspace(0, x_max, width) + 0.5 * grid_spacing, 
                       np.linspace(0, y_max, height) + 0.5 * grid_spacing)

    if cbar:
        for cb in cbar:
            cb.remove()
    

    im0 = ax[0].imshow(np.linalg.norm(u, axis=0)[::-1, :])
    cbar0 = plt.colorbar(im0, ax=ax[0])


    ax[1].imshow(p[::-1, :][::1, ::1], cmap=cmap, extent=[0,
                                                           x_max + grid_spacing, 0, y_max + grid_spacing])
    ax[1].quiver(X[::1, ::1], Y[::1, ::1], u[0,::1,::1], u[1,::1,::1], color="white",) #alpha = np.minimum(np.ones(np.shape(boundary_image)), 0.5 + np.abs(boundary_image - 1)))

    ax[2].imshow(u[0, ::-1])
    ax[3].imshow(u[1, ::-1])
    
    ax[0].set_title("u norm")
    ax[1].set_title("u and p")
    ax[2].set_title("u_x")
    ax[3].set_title("u_y")

    return [cbar0]
    
def plot_poisson(ax, x, y, boundary, cbar=None, delta_t=1, rho=1, in_=None, dy=1, dx=1, enforceBC = lambda x : x):
    for a in ax:
        a.clear()

    laplacian_phi = laplacian(x, dx, dy, boundary=boundary, numpy_=True)

    div_u_star = in_[0].cpu().detach().numpy()

    err = np.abs(laplacian_phi - div_u_star)

    x_ = x
    y_ = y

    im0 = ax[0].imshow(x_, cmap=cmap)
    im1 = ax[1].imshow(y_, cmap=cmap)

    height, width = x_.shape[0], x_.shape[1]
    X, Y = np.meshgrid(np.linspace(0, width, width) + 0.5, 
                       np.linspace(0, height, height) + 0.5)
    
    im2 = ax[2].imshow(laplacian_phi)
  
    im3 = ax[3].imshow(div_u_star)

    ax[0].set_title(f"p = {np.mean(x_):.2f} +- {np.std(x_):.2f}")
    ax[1].set_title(f"p = {np.mean(np.array(y_)):.2f} +- {np.std(np.array(y_)):.2f}")

    ax[2].set_title("Laplacian Phi")
    ax[3].set_title("Div u star")

    if cbar:
        for cb in cbar:
            cb.remove()

    cbar0 = plt.colorbar(im0, ax=ax[0])
    cbar1 = plt.colorbar(im1, ax=ax[1])
    cbar2 = plt.colorbar(im2, ax=ax[2])
    cbar3 = plt.colorbar(im3, ax=ax[3])

    return [cbar0, cbar1, cbar2, cbar3]