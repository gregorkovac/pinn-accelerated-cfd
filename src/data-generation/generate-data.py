import json
import itertools
from tqdm import tqdm
import os
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
import argparse
import random

sys.path.insert(0, "..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numeric_method.numeric_method import solve_navier_stokes
from numeric_method.geometry import get_boundary_image

# DATA_PATH = "../../../data_final_small"
# NUM_CASES = 300
# CYLINDER_SPLIT = 0
# DAM_SPLIT = 150

# TEST_SET = False

def sample_param(x):
    if isinstance(x, list):
        return random.choice(x)
    
    if isinstance(x, dict):
        return float(np.random.uniform(x["min"], x["max"]))
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Flow data generator")

    parser.add_argument("--data-path",
                        help="Where to output the data.",
                        required=True)
    parser.add_argument("--data-config-path",
                        help="Path to the dataset configuration JSON.",
                        default="./data-generation-config.json")
    parser.add_argument("--num-cases",
                        help="The number of cases to generate.",
                        default=100)
    parser.add_argument ("--cylinder-split",
                         help="At what index to start cylinder flows. All indices before this value represent pipe flows.",
                         default=20)
    parser.add_argument ("--dam-split",
                         help="At what index to start dam flows. All indices before this value represent cylinder and pipe flows.",
                         default=60)
    parser.add_argument("--test-set",
                        action="store_true",
                        help="Whether this is a training or testing set. If this flag is true, more complex data is generated.")

    args = parser.parse_args()

    data_path = args.data_path
    data_config_path = args.data_config_path
    num_cases = int(args.num_cases)
    cylinder_split = int(args.cylinder_split)
    dam_split = int(args.dam_split)
    test_set = int(args.test_set)

    if cylinder_split >= dam_split:
        print("dam_split needs to be greater than cylinder_split")
        exit(1)

    with open(data_config_path, "r") as f:
        data_config = json.load(f)

    base_case_path = data_config["base_case_path"]
    fluids = data_config["fluids"]
    if "Re" in data_config.keys():
        Re_range = data_config["Re"]
    else:
        Re_range = None
    if "inflow" in data_config.keys():
        inflow_range = data_config["inflow"]
    else:
        inflow_range = None
    dam_x_range = data_config["dam"]["x"]
    dam_y_range = data_config["dam"]["y"]
    dam_width_range = data_config["dam"]["width"]
    dam_height_range = data_config["dam"]["height"]
    cylinder_x_range = data_config["cylinder"]["x"]
    cylinder_y_range = data_config["cylinder"]["y"]
    cylinder_size_range = data_config["cylinder"]["size"]

    metadata = {
        "num_cases": num_cases,
        "cylinder_split": cylinder_split,
        "dam_split": dam_split,
        "test_set": test_set,
        "data_config": data_config
    }

    sampled_params = {}

    np.random.seed(42)

    with open(base_case_path) as f:
        base_case = json.load(f)

    if not os.path.exists(os.path.join(data_path, "flow_pics")):
        os.makedirs(os.path.join(data_path, "flow_pics"))

    for i in range(num_cases):
        print(f"[ {i} / {num_cases} ]")

        sampled_params_case = {}

        w = base_case["width"]
        h = base_case["height"]

        fluid = sample_param(fluids)
        density = fluid["density"]
        viscosity = fluid["viscosity"]

        if inflow_range is not None:
            inflow = sample_param(inflow_range)
            Re = (density * inflow * h) / viscosity
        else:
            Re = sample_param(Re_range)
            inflow = (Re * viscosity) / (density * h)

        # Convert to kinematic viscosity
        viscosity = viscosity / density

        sampled_params_case["fluid"] = fluid["fluid"]
        sampled_params_case["Re"] = Re
        sampled_params_case["inflow"] = inflow

        g = copy.deepcopy(base_case["geometry"])

        for j in range(len(g)):
            if g[j]["type"] == "inflow":
                g[j]["vel"]["x"] = inflow

        if i >= dam_split or test_set:

            dam_x = int(sample_param(dam_x_range))
            dam_y = int(sample_param(dam_y_range))
            dam_width = int(sample_param(dam_width_range))
            dam_height = int(sample_param(dam_height_range))

            dam_width = int(np.min([dam_width, (w - dam_x) - 1]))
            dam_height = int(np.min([dam_height, (h - dam_y) - 1]))

            sampled_params_case["dam_x"] = dam_x
            sampled_params_case["dam_y"] = dam_y
            sampled_params_case["dam_width"] = dam_width
            sampled_params_case["dam_height"] = dam_height

            g.append({
                "type": "wall",
                "pos": {
                    "x": dam_x,
                    "y": dam_y,
                },
                "size": {
                    "width": dam_width,
                    "height": dam_height
                },
                "dir": {
                    "x": 0,
                    "y": 1
                }
            })

        if (i >= cylinder_split and i < dam_split) or test_set:

            cylinder_x = int(sample_param(cylinder_x_range))
            cylinder_y = int(sample_param(cylinder_y_range))
            cylinder_r = int(sample_param(cylinder_size_range))
            cylinder_r = int(np.max([1, 
                                np.min([
                                    cylinder_r,
                                    cylinder_x - 2,
                                    cylinder_y - 2,
                                    (w - cylinder_x) - 2,
                                    (h - cylinder_y) - 2]
                                )]))

            sampled_params_case["cylinder_x"] = cylinder_x
            sampled_params_case["cylinder_y"] = cylinder_y
            sampled_params_case["cylinder_r"] = cylinder_r

            g.append({
                "type": "cylinder",
                "pos": {
                    "x": cylinder_x,
                    "y": cylinder_y
                },
                "size": cylinder_r
            })

        sampled_params[i] = sampled_params_case

        u, p, u_star, div_u_star, phi = solve_navier_stokes(
            base_case["width"],
            base_case["height"],
            base_case["dx"],
            base_case["dy"],
            base_case["iterations"],
            base_case["time_step"],
            density,
            viscosity,
            base_case["beta"],
            g,
            dynamic_time_step=False,
            implicit_laplacian=False,
            return_div_u_star=True,
            method_type="numeric"
        )

        u_norm = np.linalg.norm(u[-1], axis=-1)

        # if len(u_star) != base_case["iterations"]:
        #     print("NAN FOUND --- NOT SAVING")
        #     continue

        u_star = np.array(u_star[:-1])
        div_u_star = np.array(div_u_star[:-1])
        phi = np.array(phi[:-1])

        boundary_mask = get_boundary_image(g, w, h)

        params = {
            "width": base_case["width"],
            "height": base_case["height"],
            "dx": base_case["dx"],
            "dy": base_case["dy"],
            "time_step": base_case["time_step"],
            "iterations": base_case["iterations"],
            "density": density,
            "viscosity": viscosity,
            "beta": base_case["beta"],
            "geometry": g
        }

        save_path = os.path.join(data_path, str(i))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(os.path.join(save_path, "div_u_star.npy"), div_u_star)
        np.save(os.path.join(save_path, "u_star.npy"), u_star)
        np.save(os.path.join(save_path, "phi.npy"), phi)

        with open(os.path.join(save_path, "params.json"), "w") as f:
            json.dump(params, f)

        plt.imshow(u_norm * (1 - boundary_mask), cmap="viridis")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(data_path, "flow_pics", f"{str(i)}.png"))


    metadata["sampled_params"] = sampled_params

    with open(os.path.join(data_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    main()