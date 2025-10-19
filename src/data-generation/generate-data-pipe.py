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

DATA_PATH = "../../data/generated_data_final/cylinder"
NUM_CASES = 1

def main():
    case_path = "../../data/cases/pipe.json"
    
    print(case_path)

    with open(case_path) as f:
        case_ = json.load(f)

    simulated_cases = []

    for i in range(NUM_CASES):

        density = float(10 ** np.random.uniform(np.log10(1e-2), np.log10(10)))
        inflow = float(10 ** np.random.uniform(np.log10(1e-1), np.log10(5)))

        L = 254 * 0.01
        Re = np.random.uniform(0.1, 100)

        viscosity = float((density * inflow * L) / Re)


        # density = 1
        # viscosity = 0.1
        # inflow = 1
        # cylinder_r = 4

        g = copy.deepcopy(case_["geometry"])

        for i in range(len(g)):
            if g[i]["type"] == "inflow":
                g[i]["vel"]["x"] = inflow
            elif g[i]["type"] == "cylinder":
                g[i]["pos"]["x"] = cylinder_x
                g[i]["pos"]["y"] = cylinder_y
                g[i]["size"] = cylinder_r


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

        u_norm = np.linalg.norm(u[-1], axis=-1)

        if len(u_star) != case_["iterations"]:
            print("NAN FOUND --- NOT SAVING")
            continue

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
            "geometry": g
        }

        simulated_cases.append({"div_u_star": div_u_star,
                                "u_star": u_star,
                                "phi": phi,
                                "params": params,
                                "u_norm": u_norm})


        
    shuffled = np.random.permutation(simulated_cases)
    train_size = int(len(simulated_cases) * 0.6)

    if not os.path.exists(os.path.join(DATA_PATH, "flow_pics")):
        os.makedirs(os.path.join(DATA_PATH, "flow_pics"))

    for i in range(len(shuffled)):
        if i < train_size:
            dset = "train"
        else:
            dset = "test"

        path = os.path.join(DATA_PATH, dset, str(i))

        if not os.path.exists(path):
            os.makedirs(path)
    
        np.save(os.path.join(path, "div_u_star.npy"), shuffled[i]["div_u_star"])
        np.save(os.path.join(path, "u_star.npy"), shuffled[i]["u_star"])
        np.save(os.path.join(path, "phi.npy"), shuffled[i]["phi"])

        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump(shuffled[i]["params"], f)

        plt.imshow(shuffled[i]["u_norm"], cmap="viridis")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_PATH, "flow_pics", f"{str(i)}_{dset}.png"))

if __name__ == "__main__":
    main()