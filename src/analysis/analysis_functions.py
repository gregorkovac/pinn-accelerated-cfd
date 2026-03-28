import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, "..")
from modeling.data import geometry_to_mtx
from derivatives import divergence

COLOR_NUMERIC = "#1b9e77"
COLOR_POISSON = "#d95f02"
COLOR_FULL = "#7570b3"

def numeric_key(filename):
    return int(os.path.splitext(filename)[0])

def load_case(results_dir):
    case_path = results_dir

    exec_times = pd.read_csv(os.path.join(case_path, "execution_times.csv"), sep=";")

    boundaries = []
    params = []

    # params = pd.read_csv(os.path.join(case_path, "cases.csv"), sep=";")
    for f in os.listdir(os.path.join(case_path, "geometry")):
        with open(os.path.join(case_path, "geometry", f)) as fp:
            c = json.load(fp)
            _, _, _, _, _, _, boundary = geometry_to_mtx(c["geometry"], c["width"], c["height"])
            boundaries.append(boundary)
            params.append(c)

    u_numeric = np.load(os.path.join(case_path, "u_numeric.npy"), allow_pickle=True)
    p_numeric = np.load(os.path.join(case_path, "p_numeric.npy"), allow_pickle=True)

    u_poisson = np.load(os.path.join(case_path, "u_poisson.npy"), allow_pickle=True)
    p_poisson = np.load(os.path.join(case_path, "p_poisson.npy"), allow_pickle=True)

    u_full = np.load(os.path.join(case_path, "u_full.npy"), allow_pickle=True)
    p_full = np.load(os.path.join(case_path, "p_full.npy"), allow_pickle=True)


    case = {}
    case["exec_times"] = exec_times
    case["params"] = params
    case["u_numeric"] = u_numeric
    case["p_numeric"] = p_numeric
    case["u_poisson"] = u_poisson
    case["p_poisson"] = p_poisson
    case["u_full"] = u_full
    case["p_full"] = p_full
    case["boundaries"] = boundaries

    # for i in range(len(boundaries)):
    #     fig, ax = plt.subplots(1, 2)

    #     u = u_numeric[i].astype(float)
    #     u = np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2)

    #     b = boundaries[i].astype(float)

    #     ax[0].imshow(u)
    #     ax[1].imshow(b)
    #     plt.show()

    #     print(params[i])

    return case

### ERROR MEASURES ###

def L1(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return np.sum(np.abs(y_pred - y_true), axis=-1) #/ (np.sum(np.abs(y_true), axis=-1) + 1e-2)

def L2(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return np.sqrt(np.sum((y_pred - y_true) ** 2, axis=-1)) #/ np.sqrt(np.sum(y_true ** 2, axis=-1) + 1e-2)

def Linf(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return np.max(np.abs((y_pred - y_true)), axis=-1) #/ (np.max(np.abs(y_true), axis=-1) + 1e-2)

def divergence_err(y, dy, dx, norm):
    div_y = divergence(u = y, dx = dx, dy = dy, boundary=None, numpy_=True)

    if norm == "2":
        return div_y ** 2
        
    return np.abs(div_y)

def compute_errors(case, norm="2"):
    if norm == "2":
        func = L2
    elif norm == "1":
        func = L1
    elif norm == "inf":
        func = Linf

    u_err_poisson = [func(case["u_numeric"][i], case["u_poisson"][i]) for i in range(len(case["u_numeric"]))]
    p_err_poisson = [func(case["p_numeric"][i][..., None], case["p_poisson"][i][..., None]) for i in range(len(case["u_numeric"]))]

    u_err_full = [func(case["u_numeric"][i], case["u_full"][i]) for i in range(len(case["u_numeric"]))]
    p_err_full = [func(case["p_numeric"][i][..., None], case["p_full"][i][..., None]) for i in range(len(case["u_numeric"]))]

    u_denom = [np.sum(func(case["u_numeric"][i], np.zeros_like(case["u_numeric"][i]))[case["boundaries"][i] != 1]) + 1e-8 for i in range(len(case["u_numeric"]))]
    p_denom = [np.sum(func(case["p_numeric"][i][..., None], np.zeros_like(case["p_numeric"][i][..., None]))[case["boundaries"][i] != 1]) + 1e-8 for i in range(len(case["u_numeric"]))]

    divergence_err_numeric = [divergence_err(case["u_numeric"][i], case["params"][i]["dy"], case["params"][i]["dx"], norm=norm) for i in range(len(case["u_numeric"]))]
    divergence_err_poisson = [divergence_err(case["u_poisson"][i], case["params"][i]["dy"], case["params"][i]["dx"], norm=norm) for i in range(len(case["u_poisson"]))]
    divergence_err_full = [divergence_err(case["u_full"][i], case["params"][i]["dy"], case["params"][i]["dx"], norm=norm) for i in range(len(case["u_full"]))]

    err = {}
    err["u_err_poisson"] = u_err_poisson
    err["p_err_poisson"] = p_err_poisson
    err["u_err_full"] = u_err_full
    err["p_err_full"] = p_err_full

    err["u_denominator"] = u_denom
    err["p_denominator"] = p_denom

    err["divergence_numeric"] = divergence_err_numeric
    err["divergence_poisson"] = divergence_err_poisson
    err["divergence_full"] = divergence_err_full

    err["boundaries"] = case["boundaries"]

    return err

def print_err_mean_std(err):
    bound = np.array(err["boundaries"])

    err_div_n = np.array(err["divergence_numeric"])
    err_div_p = np.array(err["divergence_poisson"])
    err_div_f = np.array(err["divergence_full"])

    err_u_p = np.array(err["u_err_poisson"])
    err_u_f = np.array(err["u_err_full"])

    err_p_p = np.array(err["p_err_poisson"])
    err_p_f = np.array(err["p_err_full"])

    # u_denom = np.array(err["u_denominator"])[:, None, None] * np.ones(err_u_p.shape)
    # p_denom = np.array(err["p_denominator"])[:, None, None] * np.ones(err_p_p.shape)

    u_denom = np.array(err["u_denominator"])
    p_denom = np.array(err["p_denominator"])
    
    err_u_p = np.sum(err_u_p, axis=(1, 2)) / u_denom
    err_u_f = np.sum(err_u_f, axis=(1, 2)) / u_denom

    err_p_p = np.sum(err_p_p, axis=(1, 2)) / p_denom
    err_p_f = np.sum(err_p_f, axis=(1, 2)) / p_denom

    err_div_n = err_div_n[bound != 1]
    err_div_p = err_div_p[bound != 1]
    err_div_f = err_div_f[bound != 1]

    # err_u_p = err_u_p[bound != 1]
    # err_u_f = err_u_f[bound != 1]

    # err_p_p = err_p_p[bound != 1]
    # err_p_f = err_p_f[bound != 1]


    print(f"numeric divergence = {np.mean(err_div_n)} +- {np.std(err_div_n)}")
    print()
    print(f"u poisson error = {np.mean(err_u_p)} +- {np.std(err_u_p)}")
    print(f"p poisson error = {np.mean(err_p_p)} +- {np.std(err_p_p)}")
    print(f"poisson divergence error = {np.mean(err_div_p)} +- {np.std(err_div_p)}")
    print()
    print(f"u full error = {np.mean(err_u_f)} +- {np.std(err_u_f)}")
    print(f"p full error = {np.mean(err_p_f)} +- {np.std(err_p_f)}")
    print(f"full divergence error = {np.mean(err_div_f)} +- {np.std(err_div_f)}")

#####################

### PLOTTING ###

def err_kde_plot(err, title=""):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    sns.kdeplot(err["u_err_poisson"], fill=True, label="Poisson model", ax=ax[0], color=COLOR_POISSON, common_norm=False)
    sns.kdeplot(err["u_err_full"], fill=True, label="Navier-Stokes model", ax=ax[0], color=COLOR_FULL, common_norm=False)
    ax[0].legend()
    ax[0].set_xlabel("Velocity error")
    ax[0].set_title("Velocity error")

    sns.kdeplot(err["p_err_poisson"], fill=True, label="Poisson model", ax=ax[1], color=COLOR_POISSON, common_norm=False)
    sns.kdeplot(err["p_err_full"], fill=True, label="Navier-Stokes model", ax=ax[1], color=COLOR_FULL, common_norm=False)
    ax[1].legend()
    ax[1].set_xlabel("Pressure error")
    ax[1].set_title("Pressure error")

    plt.suptitle(title, fontsize=18)
    plt.tight_layout()

    plt.show()

def time_kde_plot(times, title=""):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    sns.kdeplot(np.array(times["numeric"]), fill=True, label="Numerical method", ax=ax, color=COLOR_NUMERIC)
    sns.kdeplot(np.array(times["poisson"]), fill=True, label="Poisson model", ax=ax, color=COLOR_POISSON)
    sns.kdeplot(np.array(times["full"]), fill=True, label="Navier-Stokes model", ax=ax, color=COLOR_FULL)
    ax.legend()
    ax.set_xlabel("Execution time [s]")

    plt.suptitle(title, fontsize=18)
    plt.tight_layout()

    plt.show()

def err_pointwise_plot(err, title=""):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    ax = ax.flatten()

    bound = np.array(err["boundaries"], dtype=float)

    u_denom = np.array(err["u_denominator"])
    p_denom = np.array(err["p_denominator"])

    u_poisson = np.array(err["u_err_poisson"], dtype=float) * (1 - bound) / (u_denom[:, None, None] * np.ones(np.array(err["u_err_poisson"], dtype=float).shape))
    p_poisson = np.array(err["p_err_poisson"], dtype=float) * (1 - bound) / (p_denom[:, None, None] * np.ones(np.array(err["p_err_poisson"], dtype=float).shape))

    u_full = np.array(err["u_err_full"], dtype=float) * (1 - bound) / (u_denom[:, None, None] * np.ones(np.array(err["u_err_poisson"], dtype=float).shape))
    p_full = np.array(err["p_err_full"], dtype=float) * (1 - bound) / (p_denom[:, None, None] * np.ones(np.array(err["p_err_poisson"], dtype=float).shape))

    u_poisson = np.mean(u_poisson, axis=0)
    p_poisson = np.mean(p_poisson, axis=0)
    u_full = np.mean(u_full, axis=0)
    p_full = np.mean(p_full, axis=0)

    umin = np.min([u_poisson.min(), u_full.min()])
    umax = np.max([u_poisson.max(), u_full.max()])
    
    pmin = np.min([p_poisson.min(), p_full.min()])
    pmax = np.max([p_poisson.max(), p_full.max()])

    # u_min = u_poisson.min()
    # u_max = u_poisson.max()
    # p_min = p_poisson.min()
    # p_max = p_poisson.max()

    im0 = ax[0].imshow(u_poisson, vmin=umin, vmax=umax)
    im1 = ax[1].imshow(u_full, vmin=umin, vmax=umax)
    im2 = ax[2].imshow(p_poisson, vmin=pmin, vmax=pmax)
    im3 = ax[3].imshow(p_full, vmin=pmin, vmax=pmax)

    ax[0].set_title("Poisson PINN", fontsize=24)
    ax[1].set_title("Navier-Stokes PINN", fontsize=24)
    ax[0].annotate(f'Velocity', xy=(-0.05, 0.5), xycoords='axes fraction', 
                      rotation=90, va='center', ha='right', 
                      fontsize=24)
    
    ax[2].annotate(f'Pressure', xy=(-0.05, 0.5), xycoords='axes fraction', 
                      rotation=90, va='center', ha='right', 
                      fontsize=24)

    # cbar0 = fig.colorbar(im0, fraction=0.046, pad=0.04)
    cbar1 = fig.colorbar(im1, ax=(ax[0], ax[1]), fraction=0.046, pad=0.04)
    # cbar2 = fig.colorbar(im2, fraction=0.046, pad=0.04)
    cbar3 = fig.colorbar(im3, ax=(ax[2], ax[3]), fraction=0.046, pad=0.04)

    # cbar0.ax.tick_params(labelsize=20)
    cbar1.ax.tick_params(labelsize=20)
    # cbar2.ax.tick_params(labelsize=20)
    cbar3.ax.tick_params(labelsize=20)

    cbar1.ax.yaxis.get_offset_text().set_fontsize(20)
    cbar3.ax.yaxis.get_offset_text().set_fontsize(20)

    cbar1.ax.yaxis.get_offset_text().set_x(3.0)
    cbar3.ax.yaxis.get_offset_text().set_x(3.0)

    # cbar0.ax.set_title(r'[m/s]', fontsize=24, pad=15)
    # cbar1.ax.set_title(r'[m/s]', fontsize=24, pad=15)
    # cbar2.ax.set_title(r'[Pa]', fontsize=24, pad=15)
    # cbar3.ax.set_title(r'[Pa]', fontsize=24, pad=15)

    for j in range(4):
        ax[j].axis('off')

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(title)

    plt.show()

def err_divergence_plot(err, title=""):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    ax = ax.flatten()

    bound = np.array(err["boundaries"], dtype=float)

    div_n = np.array(err["divergence_numeric"], dtype=float)
    div_p = np.array(err["divergence_poisson"], dtype=float)
    div_f = np.array(err["divergence_full"], dtype=float)

    div_n *= (1 - bound)
    div_p *= (1 - bound)
    div_f *= (1 - bound)

    div_n = np.mean(div_n, axis=0)
    div_p = np.mean(div_p, axis=0)
    div_f = np.mean(div_f, axis=0)

    vmin = np.min([div_n.min(), div_p.min(), div_f.min()])
    vmax = np.max([div_n.max(), div_p.max(), div_f.max()])

    im0 = ax[0].imshow(div_n[:20, :20], vmin=vmin, vmax=vmax)
    im1 = ax[1].imshow(div_p[:20, :20], vmin=vmin, vmax=vmax)
    im2 = ax[2].imshow(div_f[:20, :20], vmin=vmin, vmax=vmax)

    ax[0].set_title("Numeric method", fontsize=24)
    ax[1].set_title("Poisson PINN", fontsize=24)
    ax[2].set_title("Navier-Stokes PINN", fontsize=24)

    cbar = fig.colorbar(im2, ax=(ax[0], ax[1], ax[2]), fraction=0.046, pad=0.04)

    cbar.ax.set_title('[$s^{-1}$]', fontsize=24, pad=15)

    cbar.ax.tick_params(labelsize=20)

    cbar.ax.yaxis.get_offset_text().set_fontsize(20)

    cbar.ax.yaxis.get_offset_text().set_x(3.0)

    for j in range(3):
        ax[j].axis('off')

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(title)

    plt.show()

def print_exec_times(t):
    print(f"numeric: {np.mean(t['numeric'])} +- {np.std(t['numeric'])}")
    print(f"poisson: {np.mean(t['poisson'])} +- {np.std(t['poisson'])}")
    print(f"full: {np.mean(t['full'])} +- {np.std(t['full'])}")

def plot_velocity(ax, ux, uy, color="black", title="", scale=1):

    height, width = np.shape(ux)[0], np.shape(ux)[1]
    X, Y = np.meshgrid(np.linspace(0, width, width) + 0.5, 
                    np.linspace(0, height, height) + 0.5)

    ax.quiver(X, Y, ux, uy, color=color, alpha = 1, scale=scale, scale_units='xy')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)

#####################