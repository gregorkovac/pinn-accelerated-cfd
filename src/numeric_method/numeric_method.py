import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import sys
import os
import json
import time
from torch.quantization import quantize_dynamic
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import matplotlib.pyplot as plt

import gc

sys.path.insert(0, "..")

from modeling.model import CNN
from modeling.data import geometry_to_mtx, PoissonPINNDataset, FullPINNDataset

try:
    from .derivatives import *
    from .geometry import boundary
except ImportError:
    from derivatives import *
    from geometry import boundary

def compute_A_matrix(n, m):
    print("\t -> Creating discrete Poisson matrix...")

    # Create matrix with 4 on main diagonal and -1 on diagonals
    # above and below the main diagonal
    D = 4 * sp.eye(m) - sp.eye(m, k=1) - sp.eye(m, k=-1)

    # Create a matrix with Ds on diagonal
    # A = sp.zeros((n*m, n*m))
    A = sp.lil_matrix((n*m, n*m))

    for i in tqdm(range(n), desc="     "):
        A[i*m:(i+1)*m, i*m:(i+1)*m] = D

    # Add negative identities above and below the
    # main diagonal
    A += -sp.eye(n*m, k=m) - sp.eye(n*m, k=-m)

    print("\t -> Converting to a sparse matrix...")

    A = sp.csr_matrix(A)

    print("\t -> Done!")
    print()

    return A

def compute_discrete_laplacian(n, m, dx, dy):
    print("\t -> Creating discrete Laplacian matrix...")

    # Compute scaling factor
    # h2_inv = 1.0 / (grid_spacing ** 2) 
    dx_inv = 1.0 / dx ** 2
    dy_inv = 1.0 / dx ** 2

    # Compute main diagonal
    # D = -4 * h2_inv * sp.eye(m) 
    D = -(2 * dx_inv + 2 * dy_inv) * sp.eye(m)

    # Create the sub- and super-diagonal with 1/h²
    # D += h2_inv * sp.eye(m, k=1) + h2_inv * sp.eye(m, k=-1)
    D += dx_inv * sp.eye(m, k=1) + dx_inv * sp.eye(m, k=-1)

    # Initialize laplacian
    A = sp.lil_matrix((n * m, n * m))

    # Fill diagonal blocks
    for i in range(n):
        A[i * m:(i + 1) * m, i * m:(i + 1) * m] = D

    # Add upper and lower blocks
    # A += h2_inv * sp.eye(n * m, k=m) + h2_inv * sp.eye(n * m, k=-m)
    A += dy_inv * sp.eye(n * m, k=m) + dy_inv * sp.eye(n * m, k=-m)

    print("\t -> Converting to a sparse matrix...")
    A = sp.csr_matrix(A)

    print("\t -> Done!")

    return A

def solve_discrete_poisson(u, dx, dy, A):
    """
    Solves the discrete poisson equation \nabla^2 phi = u for Phi with Conjugate gradient method

    Args:
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        grid_spacing (float) - grid spacing

    Returns:
        Phi (numpy.ndarray) - pressure change scalar field of shape (n, m)
    """

    # Extract shape of input matrix
    n, m = u.shape

    # Vectorize the input matrix and multiply by grid spacing
    # u = -u.ravel() * grid_spacing ** 2
    u = -u.ravel() * (dx * dy)

    # Solve the system
    # phi_flat = spla.spsolve(A, u)
    # phi_flat = np.linalg.solve(A, u)
    phi_flat = spla.cg(A, u, rtol=1e-6)[0]

    # Reshape the solution vector into a matrix
    phi = phi_flat.reshape((n, m))

    return phi

divisor = 1

def navier_stokes_step(u, p, delta_t, density, viscosity, beta, dx, dy, geometry, A_poisson, A_laplacian = None, poisson_model = None, iter = 0):
    """
    Performs a step of the Navier-Stokes solver

    Args:
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        p (numpy.ndarray) - pressure scalar field of shape (n, m)
        delta_t (float) - time step
        density (float) - density constant
        viscosity (float) - viscosity constant
        beta (float) - weight for pressure term
        grid_spacing (float) - grid spacing
        geometry (dict) - geometry of the scene

    Returns:
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2) in next time step
        p (numpy.ndarray) - pressure scalar field of shape (n, m) in next time step
    """

    # Handle the Laplacian term implicitly
    if A_laplacian is not None:

        u_star = u

        # Extract shape
        n, m, _ = u_star.shape

        # Compute right hand side for implicit Laplacian system
        b = (u - delta_t * v_dot_del_v(u, dx, dy) - beta * delta_t / density * del_v(p, dx, dy))

        # Reshape
        b_x = b[:, :, 0].ravel()
        b_y = b[:, :, 1].ravel()

        # Solve the system
        u_star_x = spla.spsolve(A_laplacian, b_x)
        u_star_y = spla.spsolve(A_laplacian, b_y)

        # Reshape the solution
        u_star = np.zeros((n, m, 2))
        u_star[:, :, 0] = u_star_x.reshape((n, m))
        u_star[:, :, 1] = u_star_y.reshape((n, m))
    else:
        # Compute intermediate velocity
        u_star = u - delta_t * v_dot_del_v(u, dx, dy) \
            - beta * delta_t / density * del_v(p, dx, dy) \
            + delta_t * viscosity * laplace(u, dx, dy)

    # Apply boundary
    u_star, _ = boundary(geometry, u = u_star)

    # Multiply by constant
    div_u_star = density / delta_t * del_dot_v(u_star, dx, dy)

    if poisson_model is not None:

        with torch.no_grad():
            n, m = div_u_star.shape

            device = torch.device('cpu')

            u_star_ = torch.from_numpy(u_star).float().to(device).permute(2, 0, 1)

            grid_x, grid_y = torch.meshgrid(
                    torch.linspace(0, m * dx, m),
                    torch.linspace(0, n * dy, n),
                    indexing="ij"
                )
            coords = torch.stack([grid_y, grid_x], dim=0)

            u_dirichlet, u_dirichlet_mask, u_neumann, p_dirichlet, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, m, n)

            p_neumann = torch.tensor(p_neumann, dtype=torch.float32).permute(2, 0, 1)
            p_dirichlet = torch.tensor(p_dirichlet, dtype=torch.float32).unsqueeze(0)
            p_dirichlet_mask = torch.tensor(p_dirichlet_mask, dtype=torch.float32).unsqueeze(0)

            u_star_coords = torch.cat([u_star_, 
                                       coords, 
                                       p_neumann, 
                                       p_dirichlet,
                                       p_dirichlet_mask], dim=0).unsqueeze(0)

            # global divisor

            phi = poisson_model(u_star_coords).view(n, m).cpu().numpy()

    else:
        phi = solve_discrete_poisson(div_u_star, dx, dy, A=A_poisson)

    _, phi = boundary(geometry, p = phi)

    # Update veclocity
    u = u_star - delta_t / density * del_v(phi, dx, dy)

    # Update pressure
    p = phi + beta * p

    # Apply the boundary conditions
    u, _ = boundary(geometry, u = u)

    return u, p, u_star, div_u_star, phi

def solve_navier_stokes(width, height, dx, dy, max_iter, time_step, density, viscosity, beta, geometry, dynamic_time_step = False, implicit_laplacian = False, return_div_u_star = False, method_type = "numeric", model_path = None):
    """
    Numerically solves the Navier-Stokes equations for a given case

    Args:
        width (int) - grid width in number of points
        height (int) - grid height in number of points
        grid_spacing (float) - grid spacing
        max_iter (int) - number of iterations
        time_step (float) - time step
        density (float) - density constant
        viscosity (float) - viscosity constant
        beta (float) - weight for pressure term
        geometry (dict) - geometry of the scene
        dynamic_time_step (bool) - wether to use a dynamic time step or not

    Returns:
        u (list) - velocity vector fields for all time steps of shape (max_iter, height, width, 2)
        p (list) - pressure scalar fields for all time steps of shape (max_iter, height width)
    """

    if method_type not in ["numeric", "poisson_model", "full_model"]:
        print("Incorrect method type. Use one of [numeric, poisson_model, full_model]")
        exit(1)

    if method_type != "numeric" and model_path == None:
        print("Model folder path not specified")
        exit(1)

    # Initialize velocity and pressure
    u = np.zeros((height, width, 2))
    p = np.zeros((height, width))

    A_poisson = compute_A_matrix(height, width).toarray()

    if implicit_laplacian:
        A_laplacian = compute_discrete_laplacian(height, width, dx, dy)

        I = sp.eye(width * height)
        A_laplacian = (I - time_step * viscosity * A_laplacian)
    else:
        A_laplacian = None

    if method_type in ["full_model", "poisson_model"]:
        with open(os.path.join(model_path, "params.json"), "r") as f:
            params = json.load(f)

        with torch.no_grad():
            if method_type == "full_model":
                dataset = FullPINNDataset(data_path = None, N = 0)
            elif method_type == "poisson_model":
                dataset = PoissonPINNDataset(data_path = None, N = 0)
                
            input_channels, _, _, output_channels = dataset.get_dims()

            model = CNN(input_channels=input_channels,
                        output_channels=output_channels,
                        num_layers=int(params["num_layers"]),
                        hidden_channels=int(params["hidden_dim"]),
                        input_width=width,
                        input_height=height,
                        enforceBC=dataset.enforceBC,
                        use_skip=True)

            device = torch.device('cpu')
            
            state_dict = torch.load(os.path.join(model_path, "model.pt"), map_location=device)

            # Remove "module." prefix if present
            if any(k.startswith('module.') for k in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '')
                    new_state_dict[new_key] = v
                state_dict = new_state_dict

            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model = torch.jit.script(model)


        input_tensor = torch.empty((height*width, 3), dtype=torch.float32, device=device)
        i_indices = torch.arange(height, device=device).unsqueeze(1).repeat(1, width).reshape(-1)
        j_indices = torch.arange(width, device=device).unsqueeze(0).repeat(height, 1).reshape(-1)

        i_indices = (i_indices / (height - 1)).float()
        j_indices = (j_indices / (width - 1)).float()

    else:
        model = None
        input_tensor = None
        i_indices = None
        j_indices = None

    # Initialize lists for results
    results_u = []
    results_p = []
    results_u_star = []
    results_div_u_star = []
    results_phi = []

    print("\t -> Iterating...")

    if method_type == "full_model":
        u_dirichlet, u_dirichlet_mask, u_neumann, p_dirichlet, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, width, height)

        u_dirichlet = torch.tensor(u_dirichlet, dtype=torch.float32)
        u_dirichlet_mask = torch.tensor(u_dirichlet_mask, dtype=torch.float32)
        u_neumann = torch.tensor(u_neumann, dtype=torch.float32)
        p_dirichlet = torch.tensor(p_dirichlet, dtype=torch.float32)
        p_dirichlet_mask = torch.tensor(p_dirichlet_mask, dtype=torch.float32)
        p_neumann = torch.tensor(p_neumann, dtype=torch.float32)

        grid_x, grid_y = torch.meshgrid(
                torch.linspace(0, width * dx, width),
                torch.linspace(0, height * dy, height),
                indexing="ij"
            )
        
        coords = torch.stack([grid_y, grid_x], dim=0)
        
        x = torch.cat([coords,
                       u_dirichlet.permute(2, 0, 1),
                       u_dirichlet_mask.permute(2, 0, 1),
                       u_neumann.permute(2, 0, 1),
                       p_dirichlet.unsqueeze(0),
                       p_dirichlet_mask.unsqueeze(0),
                       p_neumann.permute(2, 0, 1),
                    ], dim=0).unsqueeze(0)
        
        out = model(x)#.squeeze(0).detach().numpy()

        out = dataset.enforceBC(x, out).squeeze(0).detach().numpy()

        u_y, u_x, p = out[0], out[1], out[2]
        u = np.stack((u_x, u_y), axis=-1)

        u_star = np.zeros((20, 20))
        div_u_star = np.zeros((20, 20))
        phi = np.zeros((20, 20))

        for _ in range(max_iter):
            results_u.append(u)
            results_p.append(p)
            results_u_star.append(u_star)
            results_div_u_star.append(div_u_star)
            results_phi.append(phi)

    else:

        u_dirichlet, u_dirichlet_mask, u_neumann, p_dirichlet, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, width, height)

        u_dirichlet = torch.tensor(u_dirichlet, dtype=torch.float32)
        u_dirichlet_mask = torch.tensor(u_dirichlet_mask, dtype=torch.float32)
        u_neumann = torch.tensor(u_neumann, dtype=torch.float32)
        p_dirichlet = torch.tensor(p_dirichlet, dtype=torch.float32)
        p_dirichlet_mask = torch.tensor(p_dirichlet_mask, dtype=torch.float32)
        p_neumann = torch.tensor(p_neumann, dtype=torch.float32)

        grid_x, grid_y = torch.meshgrid(
                torch.linspace(0, width * dx, width),
                torch.linspace(0, height * dy, height),
                indexing="ij"
            )
        
        coords = torch.stack([grid_y, grid_x], dim=0)

        # Iterate
        for iter in tqdm(range(max_iter), desc="     "):
        
            if method_type == "full_model":
                # time_step_ = torch.ones(height, width) * iter * time_step
                # time_step_ = time_step_.unsqueeze(0)

                # grid_x, grid_y = torch.meshgrid(
                #     torch.linspace(0, width * dx, width),
                #     torch.linspace(0, height * dy, height),
                #     indexing="ij"
                # )

                # coords = torch.stack([grid_y, grid_x], dim=0)

                # x = torch.cat([time_step_, coords], dim=0).unsqueeze(0)

                x = torch.cat([coords,
                        u_dirichlet.permute(2, 0, 1),
                        u_dirichlet_mask.unsqueeze(0),
                        u_neumann.permute(2, 0, 1),
                        p_dirichlet.unsqueeze(0),
                        p_dirichlet_mask.unsqueeze(0),
                        p_neumann.permute(2, 0, 1),
                        ], dim=0).unsqueeze(0)

                out = model(x).squeeze(0).detach().numpy()
                u_x, u_y, p = out[0], out[1], out[2]
                u = np.stack((u_x, u_y), axis=-1)

                u_star = np.zeros((20, 20))
                div_u_star = np.zeros((20, 20))
                phi = np.zeros((20, 20))
            else:
                # Perform one step of the solver
                u, p, u_star, div_u_star, phi = navier_stokes_step(u, p, time_step, density, viscosity, beta, dx, dy, geometry, A_poisson, A_laplacian, model, iter=iter)

            if dynamic_time_step:
                time_step = dx * 0.1 / np.max(u)

            # cfl = u * time_step / grid_spacing
            # cfl_percent = np.sum(cfl < 1) / (2 * width * height) * 100
            # print(f"CFL condition holds for {np.sum(cfl < 1)} / {2 * width * height} = {cfl_percent} % points")

            # if cfl_percent != 100:
            #     print(f"Adjusted time step: {time_step} -> {grid_spacing * 0.1 / np.max(u)}")
            #     time_step = grid_spacing * 0.05 / np.max(u)

            # lapl = laplace(u, grid_spacing)
            # eps = 1e-4
            # u = u + eps * lapl

            if np.isnan(np.min(u)) or np.isnan(np.min(p)):
                print("\t NaN found - stopping early")
                break
            
            # Save results
            results_u.append(u)
            results_p.append(p)
            results_u_star.append(u_star)
            results_div_u_star.append(div_u_star)
            results_phi.append(phi)

            if iter % 10 == 0:
                gc.collect()

    if return_div_u_star:
        return results_u, results_p, results_u_star, results_div_u_star, results_phi
    else:
        return results_u, results_p
