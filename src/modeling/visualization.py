import matplotlib as mpl
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
    
def plot_poisson(ax, x, y, boundary):
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()

    ax[0].imshow(x, cmap=cmap)
    ax[1].imshow(y, cmap=cmap)
    ax[2].imshow(np.linalg.norm(boundary, axis=2).round(), cmap=cmap)