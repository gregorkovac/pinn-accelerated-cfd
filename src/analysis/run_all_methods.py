import os
import json
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
import copy

sys.path.insert(0, "..")

from numeric_method.numeric_method import solve_navier_stokes

CASE_DIR = "./cases"
# POISSON_MODEL_PATH = "../../models/poisson/2025-09-17_14_48"
POISSON_MODEL_PATH = "../../models/poisson/2025-09-23_18_15"
# FULL_MODEL_PATH = "../../models/full/2025-09-17_15_02"
FULL_MODEL_PATH = "../../models/full/2025-09-24_16_09"
SAMPLES_PER_CASE = 50

def run_case(case, method_type, model_path=""):
    start_time = time.time()

    u, p = solve_navier_stokes(
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
            return_div_u_star=False,
            method_type=method_type,
            model_path=model_path
        )
    
    end_time = time.time()

    exec_time = end_time - start_time

    return u, p, exec_time

def randomize_case(case):
    case = copy.deepcopy(case)

    # density = np.random.uniform(0.01, 1)
    # viscosity = np.random.uniform(0.01, 1)
    # inflow = np.random.uniform(0.1, 5)
    density = 1
    viscosity = 0.1
    inflow = 1

    cylinder_x = np.random.randint(3, case["width"] - 3)
    cylinder_y = np.random.randint(3, case["height"] - 3)
    cylinder_r = np.random.randint(1, round(min(case["width"], case["height"]) / 3))
    cylinder_r = max(1, min([cylinder_r, cylinder_x - 3, cylinder_y - 3, case["width"] - cylinder_x - 3, case["height"] - cylinder_y - 3]))

    # D = 2 * cylinder_r
    # target_Re = np.random.uniform(10, 20)
    # inflow = (target_Re * viscosity) / (density * D)

    case["density"] = density
    case["viscosity"] = viscosity
    
    for i in range(len(case["geometry"])):
        if case["geometry"][i]["type"] == "inflow":
            case["geometry"][i]["vel"]["x"] = inflow
        elif case["geometry"][i]["type"] == "cylinder":
            case["geometry"][i]["pos"]["x"] = cylinder_x
            case["geometry"][i]["pos"]["y"] = cylinder_y
            case["geometry"][i]["size"] = cylinder_r

    return case, density, viscosity, inflow, cylinder_x, cylinder_y, cylinder_r

def main():
    np.random.seed(0)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M")
    results_dir = os.path.join("./results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    for case_name in os.listdir(CASE_DIR):

        u_numeric = []
        p_numeric = []
        u_poisson = []
        p_poisson = []
        u_full = []
        p_full = []

        exec_time_numeric = []
        exec_time_poisson = []
        exec_time_full = []

        case_params = {"density": [],
                       "viscosity": [],
                       "inflow": [],
                       "cylinder_x": [],
                       "cylinder_y": [],
                       "cylinder_r": []}
        
        with open(os.path.join(CASE_DIR, case_name)) as f:
            base_case = json.load(f)
        
        case_dir = os.path.join(results_dir, case_name.split(".")[0])
        geometry_dir = os.path.join(case_dir, "geometry")
        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(geometry_dir, exist_ok=True)

        for samp in range(SAMPLES_PER_CASE):

            case, density, viscosity, inflow, cylinder_x, cylinder_y, cylinder_r = randomize_case(base_case)
            
            case_params["density"].append(density)
            case_params["viscosity"].append(viscosity)
            case_params["inflow"].append(inflow)
            case_params["cylinder_x"].append(cylinder_x)
            case_params["cylinder_y"].append(cylinder_y)
            case_params["cylinder_r"].append(cylinder_r)

            print("-----------------")
            print(case)

            print("NUMERIC")
            u_n, p_n, exec_time_n = run_case(case, method_type="numeric")

            print("POISSON")
            u_p, p_p, exec_time_p = run_case(case, method_type="poisson_model", model_path=POISSON_MODEL_PATH)

            print("FULL")
            u_f, p_f, exec_time_f = run_case(case, method_type="full_model", model_path=FULL_MODEL_PATH)

            if len(u_n) != case["iterations"] or len(u_p) != case["iterations"] or len(u_f) != case["iterations"]:
                print("[[[[[[[[[[[[[[[ NAN - SKIPPING SAMPLE")
                continue

            print("-----------------")

            u_numeric.append(u_n[-1])
            p_numeric.append(p_n[-1])
            exec_time_numeric.append(exec_time_n)

            u_poisson.append(u_p[-1])
            p_poisson.append(p_p[-1])
            exec_time_poisson.append(exec_time_p)

            u_full.append(u_f[-1])
            p_full.append(p_f[-1])
            exec_time_full.append(exec_time_f)

            with open(os.path.join(geometry_dir, f"{samp}.json"), "w") as f:
                json.dump(case, f)

        df = pd.DataFrame(data = {"numeric": exec_time_numeric,
                                  "poisson": exec_time_poisson,
                                  "full": exec_time_full})
        df.to_csv(os.path.join(case_dir, "execution_times.csv"), sep=";", index=False)

        df = pd.DataFrame(case_params)
        df.to_csv(os.path.join(case_dir, "cases.csv"), sep=";", index=False)

        np.save(os.path.join(case_dir, "u_numeric.npy"), np.array(u_numeric, dtype=object))
        np.save(os.path.join(case_dir, "p_numeric.npy"), np.array(p_numeric, dtype=object))

        np.save(os.path.join(case_dir, "u_poisson.npy"), np.array(u_poisson, dtype=object))
        np.save(os.path.join(case_dir, "p_poisson.npy"), np.array(p_poisson, dtype=object))

        np.save(os.path.join(case_dir, "u_full.npy"), np.array(u_full, dtype=object))
        np.save(os.path.join(case_dir, "p_full.npy"), np.array(p_full, dtype=object))

if __name__ == "__main__":
    main()