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
    for f in sorted(os.listdir(os.path.join(case_path, "geometry")), key=numeric_key):
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

    return case

### ERROR MEASURES ###

def rmse(y_true, y_pred, bound):
    bound = 1 - bound

    return np.sqrt(np.mean(bound * (y_true - y_pred) ** 2))

def rrmse(y_true, y_pred, bound):
    return (1 - bound) * (np.sqrt(np.sum((y_true - y_pred) ** 2)) / np.sqrt(np.sum(y_true ** 2)))

def vec_rmse(y_true, y_pred, bound):
    bound = 1 - bound

    return np.sqrt(np.mean(bound * np.sum((y_true - y_pred) ** 2, axis=-1)))

# def vec_rrmse(y_true, y_pred, bound):
#     err = (np.sqrt(np.sum(np.sum((y_true - y_pred) ** 2, axis=-1))) / np.sqrt(np.sum(np.sum(y_true ** 2, axis=-1))))


#     # return (1 - bound) * (np.sqrt(np.sum(np.sum((y_true - y_pred) ** 2, axis=-1))) / np.sqrt(np.sum(np.sum(y_true ** 2, axis=-1))))

def vec_rrmse(y_true, y_pred, bound):
    y_true = y_true[bound != 1]
    y_pred = y_pred[bound != 1]

    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)

def pointwise_rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean((y_true - y_pred) ** 2, axis=0)

def pointwise_vec_rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.sum((y_true - y_pred) ** 2, axis=-1), axis=0)

def pointwise_l1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs(y_true - y_pred), axis=0)

def pointwise_vec_l1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.sum(np.abs(y_true - y_pred), axis=-1), axis=0)

def pointwise_inf(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.max(np.abs(y_true - y_pred), axis=0)

def pointwise_vec_inf(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.max(np.abs(y_true - y_pred), axis=-1), axis=0)

def divergence_err(y, dy, dx, bound):
    bound = 1 - bound

    div_y = divergence(u = y, dx = dx, dy = dy, boundary=bound, numpy_=True)

    return bound * (div_y ** 2)

    # grad_x = np.gradient(y[:,:,0], dy, dx, edge_order=2)[1]
    # grad_y = np.gradient(y[:,:,1], dy, dx, edge_order=2)[0]

    # return bound * (grad_x + grad_y) ** 2

def rinf(y_true, y_pred, bound):
    y_true = y_true[bound != 1]
    y_pred = y_pred[bound != 1]

    return np.max(np.abs(y_true - y_pred)) / np.max(np.abs(y_true))

def vec_rinf(y_true, y_pred, bound):
    y_true = y_true[bound != 1]
    y_pred = y_pred[bound != 1]

    return np.max(np.max(np.abs(y_true - y_pred), axis=-1)) / np.max(np.max(np.abs(y_true), axis=-1))

def r1(y_true, y_pred, bound):
    y_true = y_true[bound != 1]
    y_pred = y_pred[bound != 1]

    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def vec_r1(y_true, y_pred, bound):
    y_true = y_true[bound != 1]
    y_pred = y_pred[bound != 1]
    return np.sum(np.sum(np.abs(y_true - y_pred), axis=-1)) / np.sum(np.sum(np.abs(y_true), axis=-1))

def compute_errors(case, norm="2"):
    if norm == "2":
        func = rrmse
        vec_func = vec_rrmse
        pointwise_func = pointwise_rmse
        pointwise_vec_func = pointwise_vec_rmse
    elif norm == "1":
        func = r1
        vec_func = vec_r1
        pointwise_func = pointwise_l1
        pointwise_vec_func = pointwise_vec_l1
    elif norm == "inf":
        func = rinf
        vec_func = vec_rinf
        pointwise_func = pointwise_inf
        pointwise_vec_func = pointwise_vec_inf

    u_err_poisson = [vec_func(case["u_numeric"][i], case["u_poisson"][i], case["boundaries"][i]) for i in range(len(case["u_numeric"]))]
    p_err_poisson = [func(case["p_numeric"][i], case["p_poisson"][i], case["boundaries"][i]) for i in range(len(case["u_numeric"]))]

    u_err_full = [vec_func(case["u_numeric"][i], case["u_full"][i], case["boundaries"][i]) for i in range(len(case["u_numeric"]))]
    p_err_full = [func(case["p_numeric"][i], case["p_full"][i], case["boundaries"][i]) for i in range(len(case["u_numeric"]))]

    u_pointwise_poisson = pointwise_vec_func(case["u_numeric"], case["u_poisson"])
    p_pointwise_poisson = pointwise_func(case["p_numeric"], case["p_poisson"])

    u_pointwise_full = pointwise_vec_func(case["u_numeric"], case["u_full"])
    p_pointwise_full = pointwise_func(case["p_numeric"], case["p_full"])

    divergence_err_numeric = [divergence_err(case["u_numeric"][i], case["params"][i]["dy"], case["params"][i]["dx"], case["boundaries"][i]) for i in range(len(case["u_numeric"]))]
    divergence_err_poisson = [divergence_err(case["u_poisson"][i], case["params"][i]["dy"], case["params"][i]["dx"], case["boundaries"][i]) for i in range(len(case["u_poisson"]))]
    divergence_err_full = [divergence_err(case["u_full"][i], case["params"][i]["dy"], case["params"][i]["dx"], case["boundaries"][i]) for i in range(len(case["u_full"]))]

    err = {}
    err["u_err_poisson"] = u_err_poisson
    err["p_err_poisson"] = p_err_poisson
    err["u_err_full"] = u_err_full
    err["p_err_full"] = p_err_full

    err["u_pointwise_poisson"] = u_pointwise_poisson
    err["p_pointwise_poisson"] = p_pointwise_poisson
    err["u_pointwise_full"] = u_pointwise_full
    err["p_pointwise_full"] = p_pointwise_full

    err["divergence_numeric"] = divergence_err_numeric
    err["divergence_poisson"] = divergence_err_poisson
    err["divergence_full"] = divergence_err_full

    return err

def compute_errors_for_case(results, case_num, norm="2"):
    if norm == "2":
        pointwise_func = pointwise_rmse
        pointwise_vec_func = pointwise_vec_rmse
    elif norm == "1":
        pointwise_func = pointwise_l1
        pointwise_vec_func = pointwise_vec_l1
    elif norm == "inf":
        pointwise_func = pointwise_inf
        pointwise_vec_func = pointwise_vec_inf

    u_pointwise_poisson = pointwise_vec_func(results["u_numeric"][case_num][np.newaxis, ...], results["u_poisson"][case_num][np.newaxis, ...])
    p_pointwise_poisson = pointwise_func(results["p_numeric"][case_num][np.newaxis, ...], results["p_poisson"][case_num][np.newaxis, ...])

    u_pointwise_full = pointwise_vec_func(results["u_numeric"][case_num][np.newaxis, ...], results["u_full"][case_num][np.newaxis, ...])
    p_pointwise_full = pointwise_func(results["p_numeric"][case_num][np.newaxis, ...], results["p_full"][case_num][np.newaxis, ...])

    err = {}

    err["u_pointwise_poisson"] = u_pointwise_poisson
    err["p_pointwise_poisson"] = p_pointwise_poisson
    err["u_pointwise_full"] = u_pointwise_full
    err["p_pointwise_full"] = p_pointwise_full

    return err

def print_err_mean_std(err):
    print(f"numeric divergence = {np.mean(err['divergence_numeric'])} +- {np.std(err['divergence_numeric'])}")
    print()
    print(f"u poisson error = {np.mean(err['u_err_poisson'])} +- {np.std(err['u_err_poisson'])}")
    print(f"p poisson error = {np.mean(err['p_err_poisson'])} +- {np.std(err['p_err_poisson'])}")
    print(f"poisson divergence error = {np.mean(err['divergence_poisson'])} +- {np.std(err['divergence_poisson'])}")
    print()
    print(f"u full error = {np.mean(err['u_err_full'])} +- {np.std(err['u_err_full'])}")
    print(f"p full error = {np.mean(err['p_err_full'])} +- {np.std(err['p_err_full'])}")
    print(f"full divergence error = {np.mean(err['divergence_full'])} +- {np.std(err['divergence_full'])}")

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
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax = ax.flatten()

    u_poisson = err["u_pointwise_poisson"].astype(float)
    p_poisson = err["p_pointwise_poisson"].astype(float)
    u_full = err["u_pointwise_full"].astype(float)
    p_full = err["p_pointwise_full"].astype(float)

    u_min = np.min([u_poisson.min(), u_full.min()])
    u_max = np.max([u_poisson.max(), u_full.max()])
    
    p_min = np.min([p_poisson.min(), p_full.min()])
    p_max = np.max([p_poisson.max(), p_full.max()])

    # u_min = u_poisson.min()
    # u_max = u_poisson.max()
    # p_min = p_poisson.min()
    # p_max = p_poisson.max()

    im0 = ax[0].imshow(u_poisson)
    ax[0].set_title("Velocity errors (Poisson model)")

    im1 = ax[1].imshow(p_poisson)
    ax[1].set_title("Pressure errors (Poisson model)")

    im2 = ax[2].imshow(u_full)
    ax[2].set_title("Velocity errors (Full model)")

    im3 = ax[3].imshow(p_full)
    ax[3].set_title("Pressure errors (Full model)")

    fig.colorbar(im0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, fraction=0.046, pad=0.04)

    plt.tight_layout()

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