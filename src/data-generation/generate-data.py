import json
import itertools
from tqdm import tqdm
import os
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt

sys.path.insert(0, "..")

from numeric_method.numeric_method import solve_navier_stokes

DATA_PATH = "../../../data_simple_test"
NUM_CASES = 100
# CYLINDER_SPLIT = 0
# DAM_SPLIT = 50

TEST_SET = True

def main():
    case_path = "../../data/cases/pipe.json"
    
    print(case_path)

    with open(case_path) as f:
        case_ = json.load(f)

    if not os.path.exists(os.path.join(DATA_PATH, "flow_pics")):
        os.makedirs(os.path.join(DATA_PATH, "flow_pics"))

    for i in range(NUM_CASES):
        print(f"[ {i} / {NUM_CASES} ]")

        w = case_["width"]
        h = case_["height"]

        # density = float(10 ** np.random.uniform(np.log10(1e-2), np.log10(10)))
        # inflow = float(10 ** np.random.uniform(np.log10(1e-1), np.log10(5)))

        # L = (h - 2) * 0.01
        # Re = np.random.uniform(0.1, 20)

        # viscosity = float((density * inflow * L) / Re)
        # viscosity = viscosity / density

        density = 1
        viscosity = 0.01
        inflow = 1

        g = copy.deepcopy(case_["geometry"])

        for j in range(len(g)):
            if g[j]["type"] == "inflow":
                g[j]["vel"]["x"] = inflow

        dam_x = int(np.random.randint(0, w - 2))
        dam_y = int(np.random.randint(0, h - 2))
        dam_width = int(np.random.randint(4, int(w / 2)))
        dam_height = int(np.random.randint(4, int(h / 2)))

        dam_width = int(np.min([dam_width, (w - dam_x) - 1]))
        dam_height = int(np.min([dam_height, (h - dam_y) - 1]))

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

        cylinder_x = int(np.random.randint(int(w / 10), w - int(w / 10)))
        cylinder_y = int(np.random.randint(int(h / 10), h - int(h / 10)))
        cylinder_r = int(np.random.randint(int(h / 10) - 2, int(h / 4)))
        cylinder_r = int(np.min([
            cylinder_r,
            cylinder_x - 2,
            cylinder_y - 2,
            (w - cylinder_x) - 2,
            (h - cylinder_y) - 2]
        ))

        g.append({
            "type": "cylinder",
            "pos": {
                "x": cylinder_x,
                "y": cylinder_y
            },
            "size": cylinder_r
        })

        # delta_t = case_["dx"] / 100
        # delta_t = case_["time_step"]

        u, p, u_star, div_u_star, phi = solve_navier_stokes(
            case_["width"],
            case_["height"],
            case_["dx"],
            case_["dy"],
            case_["iterations"],
            case_["time_step"],
            density,
            viscosity,
            case_["beta"],
            g,
            dynamic_time_step=False,
            implicit_laplacian=True,
            return_div_u_star=True,
            method_type="numeric"
        )

        u_norm = np.linalg.norm(u[-1], axis=-1)

        # if len(u_star) != case_["iterations"]:
        #     print("NAN FOUND --- NOT SAVING")
        #     continue

        u_star = u_star[:-1]
        div_u_star = div_u_star[:-1]
        phi = phi[:-1]

        params = {
            "width": case_["width"],
            "height": case_["height"],
            "dx": case_["dx"],
            "dy": case_["dy"],
            "time_step": case_["time_step"],
            "iterations": case_["iterations"],
            "density": density,
            "viscosity": viscosity,
            "beta": case_["beta"],
            "geometry": g
        }

        save_path = os.path.join(DATA_PATH, str(i))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(os.path.join(save_path, "div_u_star.npy"), div_u_star)
        np.save(os.path.join(save_path, "u_star.npy"), u_star)
        np.save(os.path.join(save_path, "phi.npy"), phi)

        with open(os.path.join(save_path, "params.json"), "w") as f:
            json.dump(params, f)

        plt.imshow(u_norm, cmap="viridis")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_PATH, "flow_pics", f"{str(i)}.png"))

if __name__ == "__main__":
    main()