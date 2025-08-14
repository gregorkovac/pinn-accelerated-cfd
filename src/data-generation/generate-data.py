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
        grid_spacing_all = case["params"]["grid_spacing"]
        density_all = case["params"]["density"]
        viscosity_all = case["params"]["viscosity"]
        inflow_all = case["params"]["inflow"]

        # Compute cartesian product of parameters
        params_prod = itertools.product(*[grid_spacing_all, density_all, viscosity_all, inflow_all])
        
        all_combs = len(grid_spacing_all) * len(density_all) * len(viscosity_all) * len(inflow_all)

        for params in tqdm(params_prod, total=all_combs):
            grid_spacing, density, viscosity, inflow = params
            
            g = case_["geometry"]
            g[0]['vel']['x'] = inflow

            delta_t = grid_spacing / 100

            u, p, u_star, div_u_star, phi = solve_navier_stokes(
                case_["width"],
                case_["height"],
                grid_spacing,
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

            with open(os.path.join(path, "params.json"), "w") as f:
                json.dump(params, f)

            case_count += 1

if __name__ == "__main__":
    main()