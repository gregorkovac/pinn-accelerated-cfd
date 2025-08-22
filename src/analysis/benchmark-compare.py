import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

sys.path.insert(0, "..")

from numeric_method.numeric_method import solve_navier_stokes

def plot_flow(ax, u, dx, dy):

    width, height = np.shape(u)[1], np.shape(u)[0]
    x_max = (width - 1) * dx
    y_max = (height - 1) * dy

    ax.clear()
    X, Y = np.meshgrid(np.linspace(0, x_max, width) + 0.5 * dx, 
                       np.linspace(0, y_max, height) + 0.5 * dy)

    ax.quiver(X, Y, u[:,:,0], u[:,:,1], color="black")

    ax.set_aspect(dx / dy)

def plot_err(ax, u, dx, dy):
    ax.clear()
    extent = [0, u.shape[1] * dx, 0, u.shape[0] * dy]
    ax.imshow(u, extent=extent, aspect=dx/dy)

def update_animation(i, ax, u, dx, dy, frame_step = 10, title_note=""):

    i = (i * frame_step) % np.shape(u)[0]

    plot_flow(ax, u[i], dx, dy)
    ax.set_title(f"Frame {i} / {np.shape(u)[0]}{title_note}")

def update_err_animation(i, ax, u, dx, dy, frame_step = 10, title_note=""):

    i = (i * frame_step) % np.shape(u)[0]

    plot_err(ax, u[i], dx, dy)
    ax.set_title(f"{np.mean(u[i])} +- {np.std(u[i])}")

def plot_flow_animation(u_base, u, err, dx, dy, frame_step=10):
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))

    anim = animation.FuncAnimation(
        fig, update_animation, fargs=(ax[0], u_base, dx, dy, frame_step),
        frames=len(u) - 1, interval=10, blit=False
    )

    anim1 = animation.FuncAnimation(
        fig, update_animation, fargs=(ax[1], u, dx, dy, frame_step),
        frames=len(u) - 1, interval=10, blit=False
    )

    anim2 = animation.FuncAnimation(
        fig, update_err_animation, fargs=(ax[2], err, dx, dy, frame_step),
        frames=len(u) - 1, interval=10, blit=False
    )

    plt.show()

BENCHAMRK_PATH = "../../data/CFDBench"

def eval_case(case_path):
    with open(os.path.join(case_path, "case.json")) as f:
        case = json.load(f)

    print(case)
    # exit(0)

    u_base = np.load(os.path.join(case_path, "u.npy"))
    v_base = np.load(os.path.join(case_path, "v.npy"))
    uv_base = np.stack((u_base, v_base), axis=-1)

    with open(os.path.join(case_path, "../../geometry.json")) as f:
        geometry = json.load(f)

    iters, height, width = u_base.shape

    iters = 300
    uv_base = uv_base[:iters]

    # grid_spacing = case["width"] / width
    dx = (np.abs(case["x_min"]) + case["radius"] + np.abs(case["x_max"])) / width
    dy = (np.abs(case["y_min"]) + case["radius"] + np.abs(case["y_max"])) / height

    print(dx)
    print(dy)

    # cylindex_x = (np.abs(case["x_min"]) + case["radius"] / 2) / grid_spacing
    # cylindex_y = (np.abs(case["y_min"]) + case["radius"] / 2) / grid_spacing

    cylinder_y = (np.abs(case["y_min"]) + case["radius"] / 2) / dy
    cylinder_x = (np.abs(case["x_min"]) + case["radius"] / 2) / dx

    for i in range(len(geometry)):
        if geometry[i]["type"] == "cylinder":
            geometry[i]["pos"]["x"] = cylinder_x
            geometry[i]["pos"]["y"] = cylinder_y
            geometry[i]["size"] = (case["radius"] * 2) / dx
        elif geometry[i]["type"] == "inflow":
            geometry[i]["vel"]["x"] = case["vel_in"]

    u, p, u_star, div_u_star, phi = solve_navier_stokes(
        width,
        height,
        dx,
        dy,
        iters,
        1 / iters,
        case["density"],
        case["viscosity"],
        0,
        geometry,
        dynamic_time_step=True,
        implicit_laplacian=True,
        return_div_u_star=True,
        method_type="numeric"
    )

    err = np.linalg.norm(np.array(u) - uv_base, axis=-1)

    plot_flow_animation(uv_base, u, err, dx, dy, frame_step=10)

def main():
    eval_case(os.path.join(BENCHAMRK_PATH, "cylinder/bc/case0010"))

if __name__ == "__main__":
    main()