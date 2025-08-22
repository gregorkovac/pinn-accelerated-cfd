import json
import itertools
from tqdm import tqdm
import os
import numpy as np
import sys

sys.path.insert(0, "..")

from numeric_method.numeric_method import solve_navier_stokes

DATA_PATH = "../../data/generated_data"

def main():
    # Open cases file
    with open("cases.json") as f:
        cases = json.load(f)

    # Initialize generated data count
    case_count = 0
    
    # Iterate through cases
    for case in cases:
        # Read case path
        case_path = case["path"]
        
        print(case_path)

        with open(case_path) as f:
            case_ = json.load(f)

        # Read params
        density_min, density_max = case["params"]["density"]
        viscosity_min, viscosity_max = case["params"]["viscosity"]
        inflow_min, inflow_max = case["params"]["inflow"]
        cylinder_x_min, cylinder_x_max = case["params"]["cylinder_x"]
        cylinder_y_min, cylinder_y_max = case["params"]["cylinder_y"]
        cylinder_r_min, cylinder_r_max = case["params"]["cylinder_r"]

        # Compute cartesian product of parameters
        # params_prod = itertools.product(*[grid_spacing_all, density_all, viscosity_all, inflow_all])
        
        # all_combs = len(grid_spacing_all) * len(density_all) * len(viscosity_all) * len(inflow_all)

        for _ in range(case["num_cases"]):
            # grid_spacing, density, viscosity, inflow = params
            
            density = np.random.uniform(density_min, density_max)
            viscosity = np.random.uniform(viscosity_min, viscosity_max)
            inflow = np.random.uniform(inflow_min, inflow_max)
            cylinder_x = np.random.randint(cylinder_x_min, cylinder_x_max)
            cylinder_y = np.random.randint(cylinder_y_min, cylinder_y_max)
            cylinder_r = min([np.random.randint(cylinder_r_min, cylinder_r_max), cylinder_x, cylinder_y, case_["width"] - cylinder_x, case_["height"] - cylinder_y])

            print("Sampled params:")
            print(f"density = {density}")
            print(f"viscosity = {viscosity}")
            print(f"inflow = {inflow}")
            print(f"cylinder_x = {cylinder_x}")
            print(f"cylinder_y = {cylinder_y}")
            print(f"cylinder_r = {cylinder_r}")
            print()
            print()

            g = case_["geometry"]
            g[0]['vel']['x'] = inflow

            delta_t = case_["dx"] / 100

            u, p, u_star, div_u_star, phi = solve_navier_stokes(
                case_["width"],
                case_["height"],
                case_["dx"],
                case_["dy"],
                case_["iterations"],
                delta_t,
                density,
                viscosity,
                case_["beta"],
                g,
                dynamic_time_step=False,
                implicit_laplacian=True,
                return_div_u_star=True,
                method_type="numeric"
            )

            path = os.path.join(DATA_PATH, f"{case_count}")

            if not os.path.exists(path):
                 os.makedirs(path)

            np.save(os.path.join(path, "div_u_star.npy"), div_u_star)
            np.save(os.path.join(path, "u_star.npy"), u_star)
            np.save(os.path.join(path, "phi.npy"), phi)

            params = {
                "width": case_["width"],
                "height": case_["height"],
                "dx": case_["dx"],
                "dy": case_["dy"],
                "time_step": case_["time_step"],
                "iterations": case_["iterations"],
                "density": density,
                "viscosity": viscosity,
                "cylinder_x": cylinder_x,
                "cylinder_y": cylinder_y,
                "cylinder_r": cylinder_r,
                "geometry": g
            }

            with open(os.path.join(path, "params.json"), "w") as f:
                json.dump(params, f)

            case_count += 1

if __name__ == "__main__":
    main()