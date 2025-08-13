import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

sys.path.insert(0, "..")

from numeric_method.numeric_method import solve_navier_stokes

def plot_flow(ax, u, grid_spacing):

    width, height = np.shape(u)[1], np.shape(u)[0]
    x_max = (width - 1) * grid_spacing
    y_max = (height - 1) * grid_spacing

    ax.clear()
    X, Y = np.meshgrid(np.linspace(0, x_max, width) + 0.5 * grid_spacing, 
                       np.linspace(0, y_max, height) + 0.5 * grid_spacing)

    ax.quiver(X, Y, u[:,:,0], u[:,:,1], color="black")

def plot_err(ax, u, grid_spacing):
    ax.clear()
    ax.imshow(u)

def update_animation(i, ax, u, grid_spacing, frame_step = 10, title_note=""):

    i = (i * frame_step) % np.shape(u)[0]

    plot_flow(ax, u[i], grid_spacing)
    ax.set_title(f"Frame {i} / {np.shape(u)[0]}{title_note}")

def update_err_animation(i, ax, u, grid_spacing, frame_step = 10, title_note=""):

    i = (i * frame_step) % np.shape(u)[0]

    plot_err(ax, u[i], grid_spacing)
    ax.set_title(f"{np.mean(u[i])} +- {np.std(u[i])}")

def plot_flow_animation(u_base, u, err, grid_spacing, frame_step=10):
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))

    anim = animation.FuncAnimation(
        fig, update_animation, fargs=(ax[0], u_base, grid_spacing, frame_step),
        frames=len(u) - 1, interval=10, blit=False
    )

    anim1 = animation.FuncAnimation(
        fig, update_animation, fargs=(ax[1], u, grid_spacing, frame_step),
        frames=len(u) - 1, interval=10, blit=False
    )

    anim2 = animation.FuncAnimation(
        fig, update_err_animation, fargs=(ax[2], err, grid_spacing, frame_step),
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

    iters = 100
    uv_base = uv_base[:iters]

    # grid_spacing = case["width"] / width
    grid_spacing = (np.abs(case["x_min"]) + case["radius"] + np.abs(case["x_max"])) / width

    # cylindex_x = (np.abs(case["x_min"]) + case["radius"] / 2) / grid_spacing
    # cylindex_y = (np.abs(case["y_min"]) + case["radius"] / 2) / grid_spacing

    cylinder_y = 32
    cylinder_x = (np.abs(case["x_min"]) + case["radius"] / 2) / grid_spacing

    for i in range(len(geometry)):
        if geometry[i]["type"] == "cylinder":
            geometry[i]["pos"]["x"] = cylinder_x
            geometry[i]["pos"]["y"] = cylinder_y
            geometry[i]["size"] = (case["radius"] * 2) / grid_spacing
        elif geometry[i]["type"] == "inflow":
            geometry[i]["vel"]["x"] = case["vel_in"]

    u, p, u_star, div_u_star, phi = solve_navier_stokes(
        width,
        height,
        grid_spacing,
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

    plot_flow_animation(uv_base, u, err, grid_spacing=0.1, frame_step=10)

def main():
    eval_case(os.path.join(BENCHAMRK_PATH, "cylinder/bc/case0010"))

if __name__ == "__main__":
    main()