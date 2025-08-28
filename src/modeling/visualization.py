import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

cmap = mpl.cm.Blues(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[10:,:-1])

def plot_full(ax, x, grid_spacing, boundary_image = np.zeros((20, 20))):

    u = x[:2]
    p = x[2]

    u = np.transpose(u, (1, 2, 0))

    width, height = np.shape(u)[1], np.shape(u)[0]
    x_max = (width - 1) * grid_spacing
    y_max = (height - 1) * grid_spacing

    ax.clear()
    X, Y = np.meshgrid(np.linspace(0, x_max, width) + 0.5 * grid_spacing, 
                       np.linspace(0, y_max, height) + 0.5 * grid_spacing)
    # ax.contourf(X, Y, p, cmap=cmap)
    ax.imshow(p[::-1, :], cmap=cmap, extent=[0, x_max + grid_spacing, 
                                    0, y_max + grid_spacing])
    ax.quiver(X, Y, u[:,:,0], u[:,:,1], color="white", alpha = np.minimum(np.ones(np.shape(boundary_image)), 0.5 + np.abs(boundary_image - 1)))

    boundary_image = boundary_image[::-1, :]
    ax.imshow(boundary_image, cmap="Greys", extent=[0, x_max + grid_spacing, 
                                    0, y_max + grid_spacing], alpha=0.5*boundary_image)
    
def plot_poisson(ax, x, y, boundary, cbar=None):
    for a in ax:
        a.clear()

    im0 = ax[0].imshow(x, cmap=cmap)
    im1 = ax[1].imshow(y, cmap=cmap)
    im2 = ax[2].imshow(boundary, cmap=cmap)

    if cbar:
        for cb in cbar:
            cb.remove()

    cbar0 = plt.colorbar(im0, ax=ax[0])
    cbar1 = plt.colorbar(im1, ax=ax[1])
    cbar2 = plt.colorbar(im2, ax=ax[2])

    return [cbar0, cbar1, cbar2]