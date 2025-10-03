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

CASE_DIR = "../../data/generated_data_simple/test"
# POISSON_MODEL_PATH = "../../models/poisson/2025-09-17_14_48"
POISSON_MODEL_PATH = "../../models/poisson/2025-10-02_11_17"
# FULL_MODEL_PATH = "../../models/full/2025-09-17_15_02"
FULL_MODEL_PATH = "../../models/full/2025-09-29_11_53"

def run_case(case, method_type, model_path=""):
    start_time = time.time()

    u, p = solve_navier_stokes(
            case["width"],
            case["height"],
            case["dx"],
            case["dy"],
            #case["iterations"],
            1000,
            case["time_step"],
            case["density"],
            case["viscosity"],
            0,
            case["geometry"],
            dynamic_time_step=False,
            implicit_laplacian=True,
            return_div_u_star=False,
            method_type=method_type,
            model_path=model_path
        )
    
    end_time = time.time()

    exec_time = end_time - start_time

    return u, p, exec_time

def main():
    np.random.seed(0)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M")
    results_dir = os.path.join("./results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

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

    geometry_dir = os.path.join(results_dir, "geometry")
    os.makedirs(geometry_dir, exist_ok=True)

    for case_name in os.listdir(CASE_DIR):
        
        with open(os.path.join(CASE_DIR, case_name, "params.json")) as f:
            case = json.load(f)
        
        # case_dir = os.path.join(results_dir, case_name)
        # os.makedirs(case_dir, exist_ok=True)

        print("-----------------")
        print(case)

        print("NUMERIC")
        u_n, p_n, exec_time_n = run_case(case, method_type="numeric")

        print("POISSON")
        u_p, p_p, exec_time_p = run_case(case, method_type="poisson_model", model_path=POISSON_MODEL_PATH)

        print("FULL")
        u_f, p_f, exec_time_f = run_case(case, method_type="full_model", model_path=FULL_MODEL_PATH)

        # if len(u_n) != case["iterations"] or len(u_p) != case["iterations"] or len(u_f) != case["iterations"]:
        if len(u_n) != 1000 or len(u_p) != 1000 or len(u_f) != 1000:
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

        with open(os.path.join(geometry_dir, f"{case_name}.json"), "w") as f:
            json.dump(case, f)

    df = pd.DataFrame(data = {"numeric": exec_time_numeric,
                                "poisson": exec_time_poisson,
                                "full": exec_time_full})
    df.to_csv(os.path.join(results_dir, "execution_times.csv"), sep=";", index=False)

    df = pd.DataFrame(case_params)
    df.to_csv(os.path.join(results_dir, "cases.csv"), sep=";", index=False)

    np.save(os.path.join(results_dir, "u_numeric.npy"), np.array(u_numeric, dtype=object))
    np.save(os.path.join(results_dir, "p_numeric.npy"), np.array(p_numeric, dtype=object))

    np.save(os.path.join(results_dir, "u_poisson.npy"), np.array(u_poisson, dtype=object))
    np.save(os.path.join(results_dir, "p_poisson.npy"), np.array(p_poisson, dtype=object))

    np.save(os.path.join(results_dir, "u_full.npy"), np.array(u_full, dtype=object))
    np.save(os.path.join(results_dir, "p_full.npy"), np.array(p_full, dtype=object))

if __name__ == "__main__":
    main()