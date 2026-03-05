from tqdm import tqdm
# import numpy as np
import numpy as np
import json
from matplotlib import rc
import matplotlib.pyplot as plt
import sys
import copy
import json
import os

from numeric_method import solve_navier_stokes
from visualizations import plot_flow, plot_flow_animation, plot_navier_stokes_error, plot_poisson_animation
from misc import navier_stokes_error

def main(case_name = None):
    if case_name == None:
        case_name = "examples/pipe.json"
    
    with open(case_name) as f:
        case = json.load(f)

    rc("font", family="serif")

    u, p, u_star, div_u_star, phi = solve_navier_stokes(
        case["width"],
        case["height"],
        case["dx"],
        case["dy"],
        case["iterations"],
        case["time_step"],
        case["density"],
        case["viscosity"],
        case["beta"],
        case["geometry"],
        dynamic_time_step=False,
        implicit_laplacian=False,
        return_div_u_star=True,
        method_type="numeric", # numeric, poisson_model, full_model
        # model_path="../../models/full/2026-02-25_21_04",
        model_path="../../models/poisson/2026-03-04_19_43",
        poisson_alternation=-1
    )

    plot_flow_animation(u, p, case["dx"], case["geometry"], frame_step=10)
    # plot_poisson_animation(div_u_star, phi, case["grid_spacing"], case["geometry"], frame_step=100)
    # plot_navier_stokes_error(u, p, case["density"], case["viscosity"], case["time_step"], case["grid_spacing"], case["geometry"])

    # path = "./cylinder_data"
    # if not os.path.exists(path):
    #     os.makedirs(path)

    # with open(os.path.join(path, "case.json"), "w") as f:
    #     json.dump(case, f)

    # np.save(os.path.join(path, "u.npy"), u)
    # np.save(os.path.join(path, "p.npy"), p)

if __name__ == "__main__":
    args = sys.argv

    case_name = None
    if len(args) >= 2:
        case_name = args[1]

    main(case_name)