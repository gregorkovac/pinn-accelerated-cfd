import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import sys
import json
import analysis_functions
importlib.reload(analysis_functions)
from analysis_functions import *
import copy

sys.path.insert(0, "..")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from numeric_method.numeric_method import solve_navier_stokes
from modeling.data import geometry_to_mtx

plt.rcParams.update({
   "text.usetex": True,
   "font.family": "serif",
   "font.serif": ["Computer Modern Roman"],
   "axes.labelsize": 12,
   "font.size": 12,
   "legend.fontsize": 10,
   "xtick.labelsize": 10,
   "ytick.labelsize": 10,
})

BENCHAMRK_PATH = "../../../../masters-thesis/data/CFDBench/cylinder/bc"

def prepare_case(case_bench, case_nm, ux, type, changing_param, case_num=-1):
    prepared_params = {}

    if type == "tube":
        height_phys = case_bench["height"]
        width_phys = case_bench["width"]
    elif type == "cylinder":
        radius_phys = case_bench["radius"]
        height_phys = abs(case_bench["y_min"]) + 2 * radius_phys + abs(case_bench["y_max"])
        width_phys = abs(case_bench["x_min"]) + 2 * radius_phys + abs(case_bench["x_max"])

        x_phys = abs(case_bench["x_min"]) + radius_phys
        y_phys = abs(case_bench["y_min"]) + radius_phys
    
    height_px = np.shape(ux)[1]# + 2
    width_px = np.shape(ux)[2]# + 2

    # if type == "tube" and changing_param == "geo":
    #     dx = case_bench["dx"]
    #     dy = case_bench["dy"]
    #     print("!!!!")
    # else:
    dx = width_phys / width_px
    dy = height_phys / height_px

    inflow = case_bench["vel_in"]
    density = case_bench["density"]
    viscosity = case_bench["viscosity"]

    viscosity = viscosity / density

    beta = 1

    iterations = 2000

    if type == "tube" and changing_param == "geo":
        dynamic_time_step = False
        time_step = width_phys * 0.001
        # time_step = 0.001
        # time_step = 0.0001 + case_num * (0.01 - 0.0001)
    elif type == "tube" and changing_param == "prop":
        dynamic_time_step = False
        time_step = 0.00001 * case_bench["density"] / case_bench["viscosity"]
        # time_step = 0.1
    else:
        dynamic_time_step = True
        time_step = 0.1

    for gi in range(len(case_nm["geometry"])):
        if case_nm["geometry"][gi]["type"] == "inflow":
            case_nm["geometry"][gi]["vel"]["x"] = inflow
            break

    if type == "cylinder":
        radius = 2 * int(radius_phys / min(dx, dy))
        cx = int(x_phys / dx)
        cy = int(y_phys / dy)

        w = 3 * int(radius_phys / dx)
        h = 3 * int(radius_phys / dy)

        case_nm["geometry"].append({
            "type": "wall",
            "pos": {
                "x": cx - w // 2,
                "y": cy - h // 2
            },
            "size": {
                "width": w,
                "height": h
            },
            "dir": {
                "x": 0,
                "y": 1
            }
        })

    case_nm["width"] = width_px
    case_nm["height"] = height_px
    case_nm["density"] = density
    case_nm["viscosity"] = viscosity
    case_nm["beta"] = beta
    case_nm["time_step"] = time_step
    case_nm["iterations"] = iterations

    prepared_params["width_px"] = width_px
    prepared_params["height_px"] = height_px
    prepared_params["dx"] = dx
    prepared_params["dy"] = dy
    prepared_params["density"] = density
    prepared_params["viscosity"] = viscosity
    prepared_params["inflow"] = inflow
    prepared_params["beta"] = beta
    prepared_params["time_step"] = time_step
    prepared_params["iterations"] = iterations
    prepared_params["dynamic_time_step"] = dynamic_time_step

    return case_nm, prepared_params

def eval_case(case_path, case_num=-1):
    print("---------------------------")
    print(case_path)
    print("---------------------------")

    ux = np.load(os.path.join(case_path, "u.npy"))
    uy = np.load(os.path.join(case_path, "v.npy"))
    with open(os.path.join(case_path, "case.json"), "r") as f:
        case_ = json.load(f)

    if "tube" in case_path:
        case_type = "tube"
    elif "cylinder" in case_path:
        case_type = "cylinder"
    elif "cavity" in case_path:
        case_type = "cavity"
    elif "dam" in case_path:
        case_type = "dam"

    if "bc" in case_path:
        changing_param = "bc"
    elif "prop" in case_path:
        changing_param = "prop"
    elif "geo" in case_path:
        changing_param = "geo"

    with open("../../data/cases/pipe_64.json", "r") as f:
        case_nm = json.load(f)

    case_nm, prepared_params = prepare_case(case_, case_nm, ux, type=case_type, changing_param=changing_param, case_num=case_num)

    print(case_nm)

    u, p, u_star, div_u_star, phi = solve_navier_stokes(
        prepared_params["width_px"],
        prepared_params["height_px"],
        prepared_params["dx"],
        prepared_params["dy"],
        1000,
        case_nm["time_step"],
        case_nm["density"],
        case_nm["viscosity"],
        case_nm["beta"],
        case_nm["geometry"],
        dynamic_time_step=prepared_params["dynamic_time_step"],
        implicit_laplacian=True,
        return_div_u_star=True,
        method_type="numeric"
    )

    u_nm = np.transpose(u[-1], (2, 0, 1))

    # In the cylinder bc data, ux == uy
    if case_type == "cylinder" and changing_param == "bc":
        u_nm[1] = u_nm[0]

    u_baseline = np.stack([ux[-1], uy[-1]], axis=0)

    _, _, _, _, _, _, boundary = geometry_to_mtx(case_nm["geometry"], prepared_params["width_px"], prepared_params["height_px"])

    u_baseline = u_baseline.transpose(1, 2, 0)
    u_nm = u_nm.transpose(1, 2, 0)

    err_inf = Linf(u_baseline, u_nm)
    err_l1 = L1(u_baseline, u_nm)
    err_l2 =  L2(u_baseline, u_nm)

    denom_inf = Linf(u_baseline, np.zeros_like(u_baseline))
    denom_l1 = Linf(u_baseline, np.zeros_like(u_baseline))
    denom_l2 = Linf(u_baseline, np.zeros_like(u_baseline))

    results = {}

    results["case_path"] = case_path
    results["nm_mean"] = u_nm.mean()
    results["nm_std"] = u_nm.std()
    results["baseline_mean"] = u_baseline.mean()
    results["baseline_std"] = u_baseline.std()
    results["err_inf"] = err_inf
    results["err_l1"] = err_l1
    results["err_l2"] = err_l2
    results["denom_inf"] = denom_inf
    results["denom_l1"] = denom_l1
    results["denom_l2"] = denom_l2
    results["bound"] = boundary
    results["u"] = u_nm

    print(f"{case_path}: L1 mean = {err_l1[boundary != 1].sum() / (denom_l1[boundary != 1].sum() + 1e-8)}   L2 mean = {err_l2[boundary != 1].sum() / (denom_l2[boundary != 1].sum() + 1e-8)}   Linf mean = {err_inf[boundary != 1].sum() / (denom_inf[boundary != 1].sum() + 1e-8)}")

    return results

def eval_folder(folder_path, case_indices=[]):
    results = []

    case_num = 0
    for i in case_indices:
        case_path = "case" + str(i).zfill(4)
        res = eval_case(os.path.join(folder_path, case_path), case_num=case_num)

        dir_path = os.path.join("./benchmark_comparison_results_new/cylinder2", str(i))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        np.save(os.path.join(dir_path, "err_inf.npy"), res["err_inf"])
        np.save(os.path.join(dir_path, "err_l1.npy"), res["err_l1"])
        np.save(os.path.join(dir_path, "err_l2.npy"), res["err_l2"])

        np.save(os.path.join(dir_path, "denom_inf.npy"), res["denom_inf"])
        np.save(os.path.join(dir_path, "denom_l1.npy"), res["denom_l1"])
        np.save(os.path.join(dir_path, "denom_l2.npy"), res["denom_l2"])

        np.save(os.path.join(dir_path, "bound.npy"), res["bound"])

        np.save(os.path.join(dir_path, "u.npy"), res["u"])

        results.append(res)

        case_num += 1 / len(case_indices)

    df = pd.DataFrame(results)

    case_ = folder_path.split("/")[-2]
    changing_param = folder_path.split("/")[-1]

    df.to_csv(f"./benchmark_comparison_results_new/results_{case_}_{changing_param}.csv", index=False, sep=";")


def main():
    # eval_folder(folder_path="../../../CFDBench/tube/bc",
    #             case_indices=[i for i in range(1, 50)])
    # eval_folder(folder_path="../../../CFDBench/tube/geo",
    #             case_indices=[i for i in range(150, 174)])
    eval_folder(folder_path="../../../CFDBench/tube/prop",
                case_indices=[i for i in range(60, 150)])

    # eval_folder(folder_path="../../../CFDBench/cylinder/bc",
    #             case_indices=[i for i in range(21, 50)])

if __name__ == "__main__":
   main()
