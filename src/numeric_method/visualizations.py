import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

from misc import navier_stokes_error
from geometry import get_boundary_image

cmap = mpl.cm.Blues(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[10:,:-1])

def plot_flow(ax, u, p, grid_spacing, boundary_image = None, cbar = None):
    """
    Plots one image of flow

    Args:
        ax (axis)
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        p (numpy.ndarray) - pressure scalar field of shape (n, m)
        grid_spacing (float) - grid spacing
        boundary_image (numpy.ndarray) - binary image of boundary of shape (n, m)
    """

    width, height = np.shape(u)[1], np.shape(u)[0]
    x_max = (width - 1) * grid_spacing
    y_max = (height - 1) * grid_spacing

    u[boundary_image == 1, :] = 0

    ax.clear()
    X, Y = np.meshgrid(np.linspace(0, x_max, width) + 0.5 * grid_spacing, 
                       np.linspace(0, y_max, height) + 0.5 * grid_spacing)
    # ax.contourf(X, Y, p, cmap=cmap)
    # p *= (1 - boundary_image)

    # im = ax.imshow(p[::-1, :], cmap=cmap, extent=[0, x_max + grid_spacing, 
    #                                 0, y_max + grid_spacing])
    


    im = ax.imshow(np.sqrt(u[:, :, 0] ** 2 + u[:, :, 1] ** 2)[::-1, ::], extent=[0, x_max + grid_spacing, 
                                    0, y_max + grid_spacing], vmin = 0, vmax = 2)

    ax.quiver(X[::4, ::4], Y[::4, ::4], u[::4,::4,0], u[::4,::4,1], color="white", scale=40, scale_units="xy", angles="xy")
    # ax.quiver(X[::4, ::4], Y[::4, ::4], u[::4,::4,0], u[::4,::4,1], color="white", alpha = np.minimum(np.ones(np.shape(boundary_image)), 0.5 + np.abs(boundary_image - 1)))
    # im = ax.imshow(np.linalg.norm(u, axis=-1), extent=[0, x_max + grid_spacing, 
    #                                 0, y_max + grid_spacing])


    # boundary_image = boundary_image[::-1, :]
    # ax.imshow(boundary_image, cmap="Greys", extent=[0, x_max + grid_spacing, 
    #                                 0, y_max + grid_spacing], alpha=0.5*boundary_image)

    # U = -np.sin(u[..., 1])
    # V = np.cos(u[..., 0])
    # speed = np.sqrt(U**2 + V**2)
    # plt.streamplot(X, Y, u[..., 0], u[..., 1], color=speed, linewidth=1)

    if cbar is None:
        cbar = plt.colorbar(im, ax=ax)
    else:
        cbar.update_normal(im)

    return cbar


def update_animation(i, ax, u, p, grid_spacing, boundary_image = None, frame_step = 10, title_note="", cbar=None):
    """
    Plots i-th frame of animation

    Args:
        ax (axis)
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        p (numpy.ndarray) - pressure scalar field of shape (n, m)
        grid_spacing (float) - grid spacing
        boundary_image (numpy.ndarray) - binary image of boundary of shape (n, m)
    """

    i = (i * frame_step) % np.shape(u)[0]

    plot_flow(ax, u[i], p[i], grid_spacing, boundary_image, cbar)
    # ax.set_title(f"Frame {i} / {np.shape(u)[0]}{title_note}   p = {np.mean(p[i]):.2f} +- {np.std(p[i]):.2f}")
    ax.set_title(f"[{i} / {np.shape(u)[0]}{title_note}] u = {np.mean(u[i]):.2f} +- {np.std(u[i]):.2f}   p = {np.mean(p[i]):.2f} +- {np.std(p[i]):.2f}")
    # ax.set_title(f"p = {np.mean(p[i]):.2f} +- {np.std(p[i]):.2f}")

def plot_flow_animation(u, p, grid_spacing, geometry = None, frame_step=10):
    """
    Plots flow animation

    Args:
        ax (axis)
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        p (numpy.ndarray) - pressure scalar field of shape (n, m)
        grid_spacing (float) - grid spacing
        geometry (dict) - geometry dictionary
    """

    boundary_image = get_boundary_image(geometry, np.shape(u[0])[1], np.shape(u[0])[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    cbar = plot_flow(ax, u[0], p[0], grid_spacing, boundary_image)

    anim = animation.FuncAnimation(
        fig, update_animation, fargs=(ax, u, p, grid_spacing, boundary_image, frame_step, "", cbar),
        frames=len(u) - 1, interval=10, blit=False
    )
    plt.show()

def plot_navier_stokes_error(u, p, density, viscosity, time_step, grid_spacing, geometry = None):
    """
    Plots Navier-Stokes errors

    Args:
        ax (axis)
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        p (numpy.ndarray) - pressure scalar field of shape (n, m)
        density (float) - density constant
        viscosity (float) - viscosity constant
        time_step (float) - time step
        grid_spacing (float) - grid spacing
        geometry (dict) - geometry dictionary
    """

    errors = []
    for i in range(len(u)):
        u_prev = None
        u_next = None
        if i > 0:
            u_prev = u[i - 1]
        if i < len(u) - 1:
            u_next = u[i + 1]

        err = navier_stokes_error(u[i], p[i], density, viscosity, time_step, grid_spacing, u_prev, u_next, geometry)
        
        errors.append(err)

    errors = np.array(errors)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(errors[:, 0], color="black")
    ax[0].fill_between(np.linspace(0, len(errors), len(errors)), 
                       errors[:, 0] - errors[:, 1], 
                       errors[:, 0] + errors[:, 1],
                       color="black",
                       alpha=0.2)
    ax[0].set_title(f"Divergence error\n{round(errors[-1, 0], 3)} +- {round(errors[-1, 1], 3)}")
    ax[0].set_xlabel("Iteration")
    ax[0].set_xlabel("Value")

    ax[1].plot(errors[:, 2], color="black")
    ax[1].fill_between(np.linspace(0, len(errors), len(errors)), 
                       errors[:, 2] - errors[:, 3], 
                       errors[:, 2] + errors[:, 3],
                       color="black",
                       alpha=0.2)
    ax[1].set_title(f"Momentum error\n{round(errors[-1, 2], 3)} +- {round(errors[-1, 3], 3)}")
    ax[1].set_xlabel("Iteration")
    ax[1].set_xlabel("Value")

    plt.show()

def plot_poisson(ax, div_u_star, phi, grid_spacing, boundary_image = None):
    """
    Plots one image of flow

    Args:
        ax (axis)
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        p (numpy.ndarray) - pressure scalar field of shape (n, m)
        grid_spacing (float) - grid spacing
        boundary_image (numpy.ndarray) - binary image of boundary of shape (n, m)
    """

    width, height = np.shape(div_u_star)[1], np.shape(div_u_star)[0]
    x_max = (width - 1) * grid_spacing
    y_max = (height - 1) * grid_spacing

    ax[0].clear()
    ax[1].clear()
    X, Y = np.meshgrid(np.linspace(0, x_max, width) + 0.5 * grid_spacing, 
                       np.linspace(0, y_max, height) + 0.5 * grid_spacing)
    # ax.contourf(X, Y, p, cmap=cmap)
    ax[0].imshow(div_u_star[::-1, :], cmap=cmap, extent=[0, x_max + grid_spacing, 
                                    0, y_max + grid_spacing])
    ax[1].imshow(phi[::-1, :], cmap=cmap, extent=[0, x_max + grid_spacing, 
                                    0, y_max + grid_spacing])
    # ax.quiver(X, Y, u[:,:,0], u[:,:,1], color="white", alpha = np.minimum(np.ones(np.shape(boundary_image)), 0.5 + np.abs(boundary_image - 1)))

    boundary_image = boundary_image[::-1, :]
    ax[0].imshow(boundary_image, cmap="Greys", extent=[0, x_max + grid_spacing, 
                                    0, y_max + grid_spacing], alpha=0.5*boundary_image)
    ax[1].imshow(boundary_image, cmap="Greys", extent=[0, x_max + grid_spacing, 
                                    0, y_max + grid_spacing], alpha=0.5*boundary_image)

    ax[0].set_title("Div (u_star)")
    ax[1].set_title("Phi")


def update_poisson_animation(i, ax, div_u_star, phi, grid_spacing, boundary_image = None, frame_step = 10):
    """
    Plots i-th frame of animation

    Args:
        ax (axis)
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        p (numpy.ndarray) - pressure scalar field of shape (n, m)
        grid_spacing (float) - grid spacing
        boundary_image (numpy.ndarray) - binary image of boundary of shape (n, m)
    """

    i = (i * frame_step) % np.shape(div_u_star)[0]

    plot_poisson(ax, div_u_star[i], phi[i], grid_spacing, boundary_image)
    ax[0].set_title(f"Frame {i} / {np.shape(div_u_star)[0]}")

def plot_poisson_animation(div_u_start, phi, grid_spacing, geometry, frame_step=1):
    """
    Plots flow animation

    Args:
        ax (axis)
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        p (numpy.ndarray) - pressure scalar field of shape (n, m)
        grid_spacing (float) - grid spacing
        geometry (dict) - geometry dictionary
    """

    boundary_image = get_boundary_image(geometry, np.shape(div_u_start[0])[1], np.shape(div_u_start[0])[0])

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    anim = animation.FuncAnimation(
        fig, update_poisson_animation, fargs=(ax, div_u_start, phi, grid_spacing, boundary_image, frame_step),
        frames=len(div_u_start) - 1, interval=10, blit=False
    )
    plt.show()

def update_diff_animation(i, ax, u, grid_spacing, boundary_image = None, frame_step = 10, title_note="", vmin=0, vmax=1000):
    """
    Plots i-th frame of animation

    Args:
        ax (axis)
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        p (numpy.ndarray) - pressure scalar field of shape (n, m)
        grid_spacing (float) - grid spacing
        boundary_image (numpy.ndarray) - binary image of boundary of shape (n, m)
    """

    i = (i * frame_step) % np.shape(u)[0]

    u = u[i]

    width, height = np.shape(u)[1], np.shape(u)[0]
    x_max = (width - 1) * grid_spacing
    y_max = (height - 1) * grid_spacing

    ax.clear()
    X, Y = np.meshgrid(np.linspace(0, x_max, width) + 0.5 * grid_spacing, 
                       np.linspace(0, y_max, height) + 0.5 * grid_spacing)
    # ax.contourf(X, Y, p, cmap=cmap)

    im = ax.imshow(u[::-1, :], cmap="viridis", extent=[0, x_max + grid_spacing, 
                                    0, y_max + grid_spacing], vmin=vmin, vmax=vmax)
    # ax.quiver(X, Y, u[:,:,0], u[:,:,1], color="white", alpha = np.minimum(np.ones(np.shape(boundary_image)), 0.5 + np.abs(boundary_image - 1)))

    boundary_image = boundary_image[::-1, :]
    ax.imshow(boundary_image, cmap="Greys", extent=[0, x_max + grid_spacing, 
                                    0, y_max + grid_spacing], alpha=0.5*boundary_image)
    
    ax.set_title(f"Frame {i} / {np.shape(u)[0]}{title_note}")

    return im

def plot_comparison_animation(u1, p1, u2, p2, name1, name2, grid_spacing, geometry = None, frame_step=10):
    """
    Plots animation that compares two cases of velocity and pressure

    Args:
        ax (axis)
        u1 (numpy.ndarray) - first velocity vector field of shape (n, m, 2)
        p1 (numpy.ndarray) - first pressure scalar field of shape (n, m)
        u2 (numpy.ndarray) - second velocity vector field of shape (n, m, 2)
        p2 (numpy.ndarray) - second pressure scalar field of shape (n, m)
        name1 (string) - name of first case
        name2 (string) - name of second case
        grid_spacing (float) - grid spacing
        geometry (dict) - geometry dictionary
    """

    boundary_image1 = get_boundary_image(geometry, np.shape(u1[0])[1], np.shape(u1[0])[0])
    boundary_image2 = get_boundary_image(geometry, np.shape(u2[0])[1], np.shape(u2[0])[0])

    diff_u = [np.linalg.norm(u1[i] - u2[i], axis=-1) for i in range(len(u1))]
    diff_p = [(p1[i] - p2[i]) ** 2 for i in range(len(p1))]

    vmin_u = min([np.min(diff_u[i]) for i in range(len(diff_u))])
    vmax_u = min([np.max(diff_u[i]) for i in range(len(diff_u))])
    vmin_p = min([np.min(diff_p[i]) for i in range(len(diff_p))])
    vmax_p = min([np.max(diff_p[i]) for i in range(len(diff_p))])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    ax = ax.flatten()

    update_animation(0, ax[0], u1, p1, grid_spacing, boundary_image1, frame_step, title_note=f" ({name1})")
    update_animation(0, ax[1], u2, p2, grid_spacing, boundary_image2, frame_step, title_note=f" ({name2})")
    im1 = update_diff_animation(0, ax[2], diff_u, grid_spacing, boundary_image1, frame_step, title_note="u difference (vector norm)", vmin=vmin_u, vmax=vmax_u)
    im2 = update_diff_animation(0, ax[3], diff_p, grid_spacing, boundary_image1, frame_step, title_note="p difference (squared difference)", vmin=vmin_p, vmax=vmax_p)

    fig.colorbar(im1, ax=ax[2], orientation='vertical', fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[3], orientation='vertical', fraction=0.046, pad=0.04)

    def double_update(i):
        update_animation(i, ax[0], u1, p1, grid_spacing, boundary_image1, frame_step, title_note="")
        update_animation(i, ax[1], u2, p2, grid_spacing, boundary_image2, frame_step, title_note="")
        
        i_frame = (i * frame_step) % len(diff_u)
        im1.set_data(diff_u[i_frame][::-1, :])
        im2.set_data(diff_p[i_frame][::-1, :])

        ax[2].set_title(f"u difference (vector norm)")
        ax[3].set_title(f"p difference (squared difference)")

    anim = animation.FuncAnimation(
        fig, double_update,
        frames=min(len(u1), len(u2)) // frame_step,
        interval=10, blit=False
    )

    # writer = animation.FFMpegWriter(fps=24, bitrate=1800)
    # anim.save("../../anims/pipe_num_vs_pinn_mse.mp4", writer=writer)
    
    plt.show()