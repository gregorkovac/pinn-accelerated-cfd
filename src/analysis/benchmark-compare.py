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

def eval_case(case_path):
   print("---------------------------")
   print(case_path)
   print("---------------------------")

   ux = np.load(os.path.join(case_path, "u.npy"))
   uy = np.load(os.path.join(case_path, "v.npy"))
   with open(os.path.join(case_path, "case.json"), "r") as f:
       case_ = json.load(f)

   with open("../../../../masters-thesis/data/cases/pipe.json", "r") as f:
       case_nm = json.load(f)

   radius_phys = case_["radius"]
   height_phys = abs(case_["y_min"]) + 2 * radius_phys + abs(case_["y_max"])
   width_phys = abs(case_["x_min"]) + 2 * radius_phys + abs(case_["x_max"])

   # height_phys = case_["height"]
   # width_phys = case_["width"]
   inflow = case_["vel_in"]
   density = case_["density"]
   viscosity = case_["viscosity"]

   height_px = np.shape(ux)[1]
   width_px = np.shape(ux)[2]

   dy = height_phys / height_px
   dx = width_phys / width_px

   x_phys = abs(case_["x_min"]) + radius_phys
   y_phys = abs(case_["y_min"]) + radius_phys

   radius = int(radius_phys / min(dx, dy))
   x = int(x_phys / dx)
   y = int(y_phys / dy)
   case_nm["geometry"].append({"type": "cylinder",
                               "pos": {
                                   "x": x,
                                   "y": y
                               },
                               "size": radius})

   case_nm["width"] = width_px
   case_nm["height"] = height_px
   case_nm["dx"] = dx
   case_nm["dy"] = dy
   case_nm["density"] = density
   case_nm["viscosity"] = viscosity / density

   err_vec_inf = np.inf
   err_vec_l2 = np.inf
   err_vec_l1 = np.inf

   # for C in [0.001, 0.1, 1, 10]:
   for C in [0.001]:
       case_nm["time_step"] = C #np.max([dy, dx]) * C / inflow

       for gi in range(len(case_nm["geometry"])):
           if case_nm["geometry"][gi]["type"] == "inflow":
               case_nm["geometry"][gi]["vel"]["x"] = inflow
               break

       u, p, u_star, div_u_star, phi = solve_navier_stokes(
           width_px,
           height_px,
           dx,
           dy,
           # case_nm["iterations"],
           300,
           case_nm["time_step"],
           case_nm["density"],
           case_nm["viscosity"],
           0,
           case_nm["geometry"],
           dynamic_time_step=True,
           implicit_laplacian=True,
           return_div_u_star=True,
           method_type="numeric"
       )

       # u = [u[i][:, :64, :] for i in range(len(u))]

       u_nm = np.transpose(u[-1], (2, 0, 1))

       u_nm[1] = u_nm[0] # TODO: Why do the do this in the cylinder case??

       u_baseline = np.stack([ux[-1], uy[-1]], axis=0)

       _, _, _, _, _, _, boundary = geometry_to_mtx(case_nm["geometry"], width_px, height_px)

       e_vec_inf = pointwise_inf(u_baseline, u_nm)[boundary != 1]
       e_vec_l1 = pointwise_l1(u_baseline, u_nm)[boundary != 1]
       e_vec_l2 =  pointwise_rmse(u_baseline, u_nm)[boundary != 1]

       if np.mean(e_vec_inf) < np.mean(err_vec_inf):
           err_vec_inf = e_vec_inf
           err_vec_l1 = e_vec_l1
           err_vec_l2 = e_vec_l2

   results = {}

   results["case_path"] = case_path
   results["nm_mean"] = u_nm.mean()
   results["nm_std"] = u_nm.std()
   results["baseline_mean"] = u_baseline.mean()
   results["baseline_std"] = u_baseline.std()
   results["err_l1_mean"] = err_vec_l1.mean()
   results["err_l1_std"] = err_vec_l1.std()
   results["err_l2_mean"] = err_vec_l2.mean()
   results["err_l2_std"] = err_vec_l2.std()
   results["err_inf_mean"] = err_vec_inf.mean()
   results["err_inf_std"] = err_vec_inf.std()

   print(f"{case_path}: L1 error mean = {err_vec_l1.mean()}")

   return results

def main():
   results = []

   for i in range(0, 50):
       case_path = "case" + str(i).zfill(4)
       res = eval_case(os.path.join(BENCHAMRK_PATH, case_path))
       results.append(res)

   df = pd.DataFrame(results)
   df.to_csv("benchmark_comparison_results.csv", index=False, sep=";")

if __name__ == "__main__":
   main()
