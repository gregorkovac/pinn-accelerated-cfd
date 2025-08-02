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

# from poisson_solver_pinn import NN
from full_pinn import NN

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

def compute_discrete_laplacian(n, m, grid_spacing):
    print("\t -> Creating discrete Laplacian matrix...")

    # Compute scaling factor
    h2_inv = 1.0 / (grid_spacing ** 2) 

    # Compute main diagonal
    D = -4 * h2_inv * sp.eye(m) 

    # Create the sub- and super-diagonal with 1/h²
    D += h2_inv * sp.eye(m, k=1) + h2_inv * sp.eye(m, k=-1)

    # Initialize laplacian
    A = sp.lil_matrix((n * m, n * m))

    # Fill diagonal blocks
    for i in range(n):
        A[i * m:(i + 1) * m, i * m:(i + 1) * m] = D

    # Add upper and lower blocks
    A += h2_inv * sp.eye(n * m, k=m) + h2_inv * sp.eye(n * m, k=-m)

    print("\t -> Converting to a sparse matrix...")
    A = sp.csr_matrix(A)

    print("\t -> Done!")

    return A

def solve_discrete_poisson(u, grid_spacing, A):
    """
    Solves the discrete poisson equation \nabla^2 phi = u for Phi with Jacobi method

    Args:
        u (numpy.ndarray) - velocity vector field of shape (n, m, 2)
        grid_spacing (float) - grid spacing

    Returns:
        Phi (numpy.ndarray) - pressure change scalar field of shape (n, m)
    """

    # Extract shape of input matrix
    n, m = u.shape

    # Vectorize the input matrix and multiply by grid spacing
    u = -u.ravel() * grid_spacing ** 2

    # Solve the system
    # phi_flat = spla.spsolve(A, u)
    # phi_flat = np.linalg.solve(A, u)
    phi_flat = spla.cg(A, u, rtol=1e-6)[0]

    # Reshape the solution vector into a matrix
    phi = phi_flat.reshape((n, m))

    return phi

divisor = 1

def navier_stokes_step(u, p, delta_t, density, viscosity, beta, grid_spacing, geometry, A_poisson, A_laplacian = None, poisson_model = None, iter = 0, input_tensor = None, i_indices=None, j_indices=None, div_u_star_mean=None, div_u_star_std=None):
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
        b = (u - delta_t * v_dot_del_v(u, grid_spacing) - beta * delta_t / density * del_v(p, grid_spacing))

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
        u_star = u - delta_t * v_dot_del_v(u, grid_spacing) \
            - beta * delta_t / density * del_v(p, grid_spacing) \
            + delta_t * viscosity * laplace(u, grid_spacing)

    # Apply boundary
    u_star, _ = boundary(geometry, u = u_star)

    # Multiply by constant
    div_u_star = density / delta_t * del_dot_v(u_star, grid_spacing)

    if poisson_model is not None:
        # phi = poisson_model(torch.tensor(div_u_star, dtype=torch.float32))
        # phi = np.zeros(np.shape(div_u_star))
        # for i in range(len(phi)):
        #     for j in range(len(phi[0])):
        #         inp = torch.tensor(np.array([j, i, iter, div_u_star[i, j]]), dtype=torch.float32)
        #         phi_ = poisson_model(inp)
        #         phi[i, j] = phi_

        with torch.no_grad():
            n, m = div_u_star.shape
            # i_indices, j_indices = np.meshgrid(np.arange(n), np.arange(m), indexing='ij')

            # # inputs = np.stack([j_indices, i_indices, np.full_like(j_indices, iter), div_u_star], axis=-1)
            # # inputs_tensor = torch.tensor(inputs.reshape(-1, 4), dtype=torch.float32)

            # input_tensor[:, 0] = torch.from_numpy(j_indices.ravel()).to(input_tensor)
            # input_tensor[:, 1] = torch.from_numpy(i_indices.ravel()).to(input_tensor)
            # input_tensor[:, 2] = iter
            # input_tensor[:, 3] = torch.from_numpy(div_u_star.ravel()).to(input_tensor)

            # t = time.time()
            device = torch.device('cpu')
            # print(f"Get device: {time.time() - t}")

            # i_indices = torch.arange(n, device=device).unsqueeze(1).repeat(1, m)
            # j_indices = torch.arange(m, device=device).unsqueeze(0).repeat(n, 1)

            # TODO: Time step ok?

            # t = time.time()

            # input_tensor[:, 0] = j_indices
            # input_tensor[:, 1] = i_indices
            # # input_tensor[:, 2] = iter
            # # input_tensor[:, 3] = torch.tensor(div_u_star, device=device).reshape(-1)
            # # div_u_star = (div_u_star - div_u_star_mean) / div_u_star_std
            # input_tensor[:, 2].copy_(torch.from_numpy(div_u_star).to(device).reshape(-1))
            # # print(f"Set tensor: {time.time() - t}")


            # div_u_star = (div_u_star - div_u_star_mean) / div_u_star_std

            # t = time.time()

            u_star_ = torch.from_numpy(u_star).float().to(device).permute(2, 0, 1).unsqueeze(0) #.reshape(-1)
            # u_star_[..., 0] = (u_star_[..., 0] - div_u_star_mean[0]) / div_u_star_std[0]
            # u_star_[..., 1] = (u_star_[..., 1] - div_u_star_mean[1]) / div_u_star_std[1]

            # u_star_ = (u_star_ - div_u_star_mean) / div_u_star_std

            grid_x, grid_y = torch.meshgrid(
                    torch.linspace(0, 20 * grid_spacing, 20),
                    torch.linspace(0, 20 * grid_spacing, 20),
                    indexing="ij"
                )
            coords = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0)

            # grid_x, grid_y = torch.meshgrid(
            #     torch.linspace(0, 1, 20),
            #     torch.linspace(0, 1, 20),
            #     indexing="ij"
            # )

            # grid_y = grid_y.clone()

            # for j in range(20):
            #     grid_y[:, j] = grid_y[:, min(j, 20 - 1 - j)]

            # coords = torch.stack([grid_x, 1 - grid_y], dim=0)
            # coords = coords.unsqueeze(0)

            u_star_coords = torch.cat([u_star_, coords], dim=1)

            # print(u_star_.shape)

            global divisor

            phi = poisson_model(u_star_coords).view(n, m).cpu().numpy()

            if iter == 0:
                phi_nm = solve_discrete_poisson(div_u_star, grid_spacing, A=A_poisson)
                divisor = np.mean(phi_nm / (phi + 1e-15))
                print(divisor)
                # divisor = 1

            phi = phi * divisor


            # print(phi[:, -1])

            # plt.imshow(phi)
            # plt.show()

            # print(u_star_.shape)
            # exit(0)

            # phi_flat = poisson_model(torch.from_numpy(div_u_star).float().to(device).reshape(-1))
            # print(f"Run model: {time.time() - t}")

            # t = time.time()
            # phi = -phi_flat.view(n, m).cpu().numpy()
            # phi_flat = phi_flat.detach().numpy()
            # phi = phi_flat.reshape(n, m)
            # print(f"Handle phi: {time.time() - t}")
            # print()
            # print()
            # print()

        # phi = np.zeros(np.shape(phi))
    else:
        # Solve the Poisson equation
        # t = time.time()
        phi = solve_discrete_poisson(div_u_star, grid_spacing, A=A_poisson)

        # fig, ax = plt.subplots(1, 2)

        # ax[1].imshow(phi)
        # ax[0].imshow(div_u_star)
        # ax[1].set_title("Pressure")
        # ax[0].set_title("Divergence of u*")
        # plt.show()

        # print(f"Solve poisson: {time.time() - t}")

        # Apply the boundary conditions

    # phi = solve_discrete_poisson(div_u_star, grid_spacing, A=A_poisson)
    # print(f"real = {np.mean(phi)}")
    # print()

    _, phi = boundary(geometry, p = phi)

    # Update veclocity
    u = u_star - delta_t / density * del_v(phi, grid_spacing)

    # Update pressure
    p = phi + beta * p

    # Apply the boundary conditions
    u, _ = boundary(geometry, u = u)

    return u, p, u_star, div_u_star, phi

def solve_navier_stokes(width, height, grid_spacing, max_iter, time_step, density, viscosity, beta, geometry, dynamic_time_step = False, implicit_laplacian = False, return_div_u_star = False, use_poisson_model=False):
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

    # Initialize velocity and pressure
    u = np.zeros((height, width, 2))
    p = np.zeros((height, width))

    A_poisson = compute_A_matrix(height, width).toarray()

    # print(A_poisson.shape)

    # u, s, v = sp.linalg.svds(A_poisson, k=899)
    # print(len(s))
    # print(np.max(s) / np.min(s))
    # exit(0)

    if implicit_laplacian:
        A_laplacian = compute_discrete_laplacian(height, width, grid_spacing)

        I = sp.eye(width * height)
        A_laplacian = (I - time_step * viscosity * A_laplacian)
    else:
        A_laplacian = None

    if use_poisson_model:
        with open(os.path.join("..", "params.json"), "r") as f:
            params = json.load(f)

        # torch.set_num_threads(os.cpu_count())

        div_u_star_mean = params["x_mean"]
        div_u_star_std = params["x_std"]

        # div_u_star_mean = None
        # div_u_star_std = None

        with torch.no_grad():
            # poisson_model = NN(num_hidden=params["num_hidden"],
            #                     hidden_dim=params["hidden_dim"])

            poisson_model = NN(input_channels=3,
                    num_layers=int(params["num_layers"]),
                    hidden_channels=int(params["hidden_dim"]),
                    input_width=20,
                    input_height=20)

            device = torch.device('cpu')

            # poisson_model = torch.jit.load(os.path.join("..", "nn_scripted.pt"), map_location=device)
            
            state_dict = torch.load(os.path.join("..", "nn.pt"), map_location=device)

            # Remove "module." prefix if present
            if any(k.startswith('module.') for k in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '')
                    new_state_dict[new_key] = v
                state_dict = new_state_dict

            poisson_model.load_state_dict(state_dict, strict=True)
            # poisson_model = torch.compile(poisson_model)
            poisson_model.eval()
            # dummy_input = torch.randn(height * width, 4) 
            # _ = poisson_model(dummy_input)
            # poisson_model = quantize_dynamic(poisson_model, {torch.nn.Linear, torch.nn.Tanh}, dtype=torch.qint8)
            # dummy_input = torch.randn(height * width, 4) 
            # poisson_model = torch.jit.trace(poisson_model, dummy_input)
            poisson_model = torch.jit.script(poisson_model)

            # t = time.time()
            # dummy_input = torch.randn(height * width, 4) 
            # _ = poisson_model(dummy_input)
            # print(f"Time: {time.time() - t}")
            # exit(0)


            # poisson_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            # torch.quantization.prepare(poisson_model, inplace=True)
            # torch.quantization.convert(poisson_model, inplace=True)


        input_tensor = torch.empty((height*width, 3), dtype=torch.float32, device=device)
        i_indices = torch.arange(height, device=device).unsqueeze(1).repeat(1, width).reshape(-1)
        j_indices = torch.arange(width, device=device).unsqueeze(0).repeat(height, 1).reshape(-1)

        i_indices = (i_indices / (height - 1)).float()
        j_indices = (j_indices / (width - 1)).float()

    else:
        poisson_model = None
        input_tensor = None
        i_indices = None
        j_indices = None
        div_u_star_mean = None
        div_u_star_std = None

    # Initialize lists for results
    results_u = []
    results_p = []
    results_u_star = []
    results_div_u_star = []
    results_phi = []

    print("\t -> Iterating...")

    # Iterate
    for iter in tqdm(range(max_iter), desc="     "):

        if use_poisson_model:
            time_step_ = torch.ones(height, width) * iter * time_step
            time_step_ = time_step_.unsqueeze(0)

            grid_x, grid_y = torch.meshgrid(
                torch.linspace(0, width * grid_spacing, width),
                torch.linspace(0, height * grid_spacing, height),
                indexing="ij"
            )

            coords = torch.stack([grid_y, grid_x], dim=0)

            x = torch.cat([time_step_, coords], dim=0).unsqueeze(0)

            out = poisson_model(x).squeeze(0).detach().numpy()
            u_x, u_y, p = out[0], out[1], out[2]
            u = np.stack((u_x, u_y), axis=-1)

            u_star = np.zeros((20, 20))
            div_u_star = np.zeros((20, 20))
            phi = np.zeros((20, 20))
        else:
            # Perform one step of the solver
            u, p, u_star, div_u_star, phi = navier_stokes_step(u, p, time_step, density, viscosity, beta, grid_spacing, geometry, A_poisson, A_laplacian, poisson_model, iter=iter, input_tensor=input_tensor, i_indices=i_indices, j_indices=j_indices, div_u_star_mean=div_u_star_mean, div_u_star_std=div_u_star_std)

        if dynamic_time_step:
            time_step = grid_spacing * 0.1 / np.max(u)

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
