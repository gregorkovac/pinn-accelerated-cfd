import json
import itertools
from tqdm import tqdm
import os
import numpy as np
import sys

sys.path.insert(0, "..")

from numeric_method.numeric_method import solve_navier_stokes

DATA_PATH = "../../data/generated_data_final/cylinder"

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
        dam_x_min, dam_x_max = case["params"]["dam_x"]
        dam_y_min, dam_y_max = case["params"]["dam_y"]
        dam_height_min, dam_height_max = case["params"]["dam_height"]
        dam_width_min, dam_width_max = case["params"]["dam_width"]

        # Compute cartesian product of parameters
        # params_prod = itertools.product(*[grid_spacing_all, density_all, viscosity_all, inflow_all])
        
        # all_combs = len(grid_spacing_all) * len(density_all) * len(viscosity_all) * len(inflow_all)

        used_cases = []

        num_cases = case["num_cases"]
        num_test_cases = case["num_test_cases"]

        for ci in range(num_cases + num_test_cases):

            if ci >= num_cases:
                path = os.path.join(DATA_PATH, "test", f"{case_count}")
            else:
                path = os.path.join(DATA_PATH, "train", f"{case_count}")

            # grid_spacing, density, viscosity, inflow = params
            
            density = np.random.uniform(density_min, density_max)
            viscosity = np.random.uniform(viscosity_min, viscosity_max)
            # inflow = np.random.uniform(inflow_min, inflow_max)

            # case_type = np.random.choice(np.repeat(["pipe", "dam", "cylinder"], [10, 45, 45]))
            # case_type = np.random.choice(["dam", "cylinder"])
            case_type = "cylinder"

            duplicate = True
            dup_iters = 0
            while duplicate:
                duplicate = False
                if cylinder_x_min == cylinder_x_max:
                    cylinder_x = cylinder_x_min
                else:
                    cylinder_x = np.random.randint(cylinder_x_min, cylinder_x_max)

                if cylinder_y_min == cylinder_y_max:
                    cylinder_y = cylinder_y_min
                else:
                    cylinder_y = np.random.randint(cylinder_y_min, cylinder_y_max)

                if cylinder_r_min == cylinder_r_max:
                    cylinder_r = cylinder_r_min
                else:
                    cylinder_r = np.random.randint(cylinder_r_min, cylinder_r_max)

                # cylinder_r = max(1, min([cylinder_r, cylinder_x, cylinder_y, case_["width"] - cylinder_x, case_["height"] - cylinder_y]))

                if [cylinder_x, cylinder_y, cylinder_r] in used_cases:
                    duplicate = True
                    dup_iters += 1

                if dup_iters == 100:
                    print("Sampling exhausted - exiting")
                    exit(0)

            used_cases.append([cylinder_x, cylinder_y, cylinder_r])

            # D = 2 * cylinder_r
            # target_Re = np.random.uniform(10, 20)
            # inflow = (target_Re * viscosity) / (density * D)
            inflow = 1

            dam_x = np.random.randint(dam_x_min, dam_x_max)
            dam_y = np.random.randint(dam_y_min, dam_y_max)
            dam_height = min(np.random.randint(dam_height_min, dam_height_max), case_["height"] - dam_y)
            dam_width = min(np.random.randint(dam_width_min, dam_width_max), case_["width"] - dam_x)

            print("Sampled params:")
            print(f"case = {case_type}")
            print(f"density = {density}")
            print(f"viscosity = {viscosity}")
            print(f"inflow = {inflow}")
            print(f"cylinder_x = {cylinder_x}")
            print(f"cylinder_y = {cylinder_y}")
            print(f"cylinder_r = {cylinder_r}")
            print(f"dam_x = {dam_x}")
            print(f"dam_y = {dam_y}")
            print(f"dam_width = {dam_width}")
            print(f"dam_height = {dam_height}")
            print()
            print()

            g = case_["geometry"]

            for i in range(len(g)):
                if g[i]["type"] == "inflow":
                    g[i]["vel"]["x"] = inflow
                elif g[i]["type"] == "cylinder":
                    g[i]["pos"]["x"] = cylinder_x
                    g[i]["pos"]["y"] = cylinder_y
                    g[i]["size"] = cylinder_r

            if case_type != "cylinder":
                g = [g_ for g_ in g if g_.get("type") != "cylinder"]
            
            if case_type == "dam":
                g.append({
                    "type": "wall",
                    "pos": {
                        "x": dam_x,
                        "y": dam_y
                    },
                    "size": {
                        "width": dam_width,
                        "height": dam_height
                    },
                    "dir": {
                        "x": 0,
                        "y": 0
                    }
                })


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

            if not os.path.exists(path):
                 os.makedirs(path)

            if len(u_star) != case_["iterations"]:
                print("NAN FOUND --- NOT SAVING")
                continue

            # TODO: Filter out very similar cases

            u_star = u_star[:-1]
            div_u_star = div_u_star[:-1]
            phi = phi[:-1]

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
                "geometry": g
            }

            with open(os.path.join(path, "params.json"), "w") as f:
                json.dump(params, f)

            case_count += 1

if __name__ == "__main__":
    main()