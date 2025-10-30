import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import splu, factorized
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.model import CNN
from modeling.data import geometry_to_mtx, PoissonPINNDataset, FullPINNDataset

from derivatives import *

try:
    # from .derivatives import *
    # from ..derivatives import *
    from .geometry import boundary
except ImportError:
    # from derivatives import *
    from geometry import boundary

def compute_discrete_laplacian(n, m, dx, dy):
    print("\t -> Creating discrete Laplacian matrix...")

    dx_inv = 1.0 / dx ** 2
    dy_inv = 1.0 / dy ** 2

    # Main diagonal
    D = -(2 * dx_inv + 2 * dy_inv) * sp.eye(m)
    
    # y-direction neighbors
    D += dy_inv * sp.eye(m, k=1) + dy_inv * sp.eye(m, k=-1)

    A = sp.lil_matrix((n * m, n * m))

    # Fill diagonal blocks
    for i in range(n):
        A[i * m:(i + 1) * m, i * m:(i + 1) * m] = D

    # x-direction neighbors
    A += dx_inv * sp.eye(n * m, k=m) + dx_inv * sp.eye(n * m, k=-m)

    A = sp.csr_matrix(A)
    return A

def solve_discrete_poisson(u, dx, dy, A, p_dirichlet_mask):
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

    # Reshape the velocity into a vector
    u = u.ravel()

    # i = p_dirichlet_mask.ravel().astype(bool)
    # u[i] = 0

    # Solve the system
    phi_flat, info = spla.cg(A, u, rtol=1e-12, atol=1e-12)
    # print("CG info:", info)

    # Reshape the solution vector into a matrix
    phi = phi_flat.reshape((n, m))

    return phi

divisor = 1

debug_data = {}
figdbg = None
axdbg = None
keys = []
lines = {}
img = None
cbar = None

def navier_stokes_step(u, p, delta_t, density, viscosity, beta, dx, dy, geometry, A_poisson, A_laplacian = None, poisson_model = None, iter = 0, geometry_mtx=None):
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

    n, m, _ = u.shape

    _, u_dirichlet_mask, u_neumann, _, p_dirichlet_mask, p_neumann, boundary_mask = geometry_mtx

    # fig, ax = plt.subplots(1, 2, figsize=(16, 12))
    # im0 = ax[0].imshow(A_poisson.toarray())
    # im1 = ax[1].imshow(boundary_mask.ravel().reshape((64, 1)))

    # plt.colorbar(im0)
    # plt.colorbar(im1)

    # plt.show()

    # boundary_mask_p = p_dirichlet_mask.transpose(1, 0).astype(bool) | np.abs(p_neumann[...,0].transpose(1, 0)).astype(bool) | np.abs(p_neumann[...,1].transpose(1, 0)).astype(bool)
    # boundary_mask_u = u_dirichlet_mask.transpose(2, 1, 0).astype(bool) | np.abs(u_neumann.transpose(2, 1, 0)).astype(bool)
    # boundary_mask_u[0] = np.sign(boundary_mask_u[0] + boundary_mask_u[1])
    # boundary_mask_u[1] = np.sign(boundary_mask_u[1] + boundary_mask_u[0])

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(boundary_mask_u[0])
    # ax[1].imshow(boundary_mask_u[1])
    # plt.show()

    # boundary_mask[1:-1, 0] = 0
    # boundary_mask[1:-1, -1] = 0

    # plt.ioff()
    # plt.imshow(boundary_mask)
    # plt.show()

    # boundary_mask[0, 1:-1] = 0
    # boundary_mask[-1, 1:-1] = 0

    boundary_mask_p =  boundary_mask.transpose(1, 0) * 0

    boundary_mask_u = np.expand_dims(boundary_mask, axis=0)
    boundary_mask_u = np.repeat(boundary_mask_u, 2, axis=0).transpose(0, 2, 1) * 0

    # boundary_mask_u[:, 0, 1:-1] = 0
    # boundary_mask_u[:, -1, 1:-1] = 0

    # Handle the Laplacian term implicitly
    if A_laplacian is not None:

        u_star = u

        # Extract shape
        n, m, _ = u_star.shape

        # Compute the convective term
        conv_u = convective(u = u.transpose(2, 1, 0),
                          dx = dx,
                          dy = dy,
                          boundary=boundary_mask_u,
                          numpy_=True).transpose(2, 1, 0)

        # Compute the pressure gradient
        grad_p = grad(u=p.transpose(1, 0),
                      dx=dx,
                      dy=dy,
                      boundary=boundary_mask_p,
                      numpy_=True).transpose(2, 1, 0)

        # Compute right hand side for implicit Laplacian system
        b = (u - delta_t * conv_u - beta * delta_t / density * grad_p)

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

        # # Compute right hand side for implicit Laplacian system
        # b = (u - delta_t * conv_u - beta * delta_t / density * grad_p)

        # # Reshape
        # b_x = b[:, :, 0].ravel()
        # b_y = b[:, :, 1].ravel()

        # A_factored = splu(A_laplacian.tocsc())

        # # Solve systems
        # B = np.vstack([b_x, b_y]).T
        # U = A_factored.solve(B)

        # # Extract u star
        # u_star_x = U[:,0].reshape((n,m))
        # u_star_y = U[:,1].reshape((n,m))

        # # Reshape the solution
        # u_star = np.zeros((n, m, 2))
        # u_star[:, :, 1] = u_star_x.reshape((n, m))
        # u_star[:, :, 0] = u_star_y.reshape((n, m))
    else:
        # Compute convective term
        conv_u = convective(u=u.transpose(2, 1, 0),
                          dx=dx,
                          dy=dy,
                          boundary=boundary_mask_u,
                          numpy_=True).transpose(2, 1, 0)

        # Compute pressure gradient        
        grad_p = grad(u=p.transpose(1, 0),
                      dx=dx,
                      dy=dy,
                      boundary=boundary_mask_p,
                      numpy_=True).transpose(2, 1, 0)
        

        # Compute laplacian term        
        lapl_u = laplacian(u=u.transpose(2, 1, 0),
                           dx=dx,
                           dy=dy,
                           boundary=boundary_mask_u,
                           numpy_=True).transpose(2, 1, 0)

        # Compute intermediate velocity
        u_star = u - delta_t * conv_u - beta * delta_t / density * grad_p + delta_t * viscosity * lapl_u

    # Apply boundary
    u_star, _ = boundary(geometry, u = u_star)

    # Compute divergence of u star
    div = divergence(u=u_star.transpose(2, 1, 0),
                     dx=dx,
                     dy=dy,
                     boundary=boundary_mask_u,
                     numpy_=True).transpose(1, 0)


    # Scale the divergence of u star
    div_u_star = density / delta_t * div

    # div_u_star = div_u_star * boundary_mask

    # Solve Poisson's equation using a model
    if poisson_model is not None:

        with torch.no_grad():
            n, m = div_u_star.shape
            device = torch.device('cpu')

            # Convert u star to Torch
            u_star_ = torch.from_numpy(u_star).float().to(device).permute(2, 0, 1)

            # Create a coordinate grid
            grid_x, grid_y = torch.meshgrid(
                    torch.linspace(0, m * dx, m),
                    torch.linspace(0, n * dy, n),
                    indexing="ij"
                )
            coords = torch.stack([grid_y, grid_x], dim=0)

            # Obtain required boundary representations
            u_dirichlet, u_dirichlet_mask, u_neumann, p_dirichlet, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, m, n)

            # Convert boundaries to Torch
            p_neumann = torch.tensor(p_neumann, dtype=torch.float32).permute(2, 0, 1)
            p_dirichlet = torch.tensor(p_dirichlet, dtype=torch.float32).unsqueeze(0)
            p_dirichlet_mask = torch.tensor(p_dirichlet_mask, dtype=torch.float32).unsqueeze(0)

            # Construct the model's input tensor
            u_star_coords = torch.cat([u_star_, 
                                       coords, 
                                       p_neumann, 
                                       p_dirichlet,
                                       p_dirichlet_mask], dim=0).unsqueeze(0)

            # Perform a forward pass and obtain the pressure correction
            phi = poisson_model(u_star_coords).view(n, m).cpu().numpy()
    else:
        # Obtain the pressure correction by numerically solving Poisson's equatio
        phi = solve_discrete_poisson(div_u_star, dx, dy, A=A_poisson, p_dirichlet_mask=p_dirichlet_mask)


    # Apply the boundary conditions to the pressure correction
    _, phi = boundary(geometry, p = phi)

    # phi -= np.mean(phi)

    # Compute the gradient of the pressure correction
    grad_phi = grad(u=phi.transpose(1, 0),
                    dx=dx,
                    dy=dy,
                    boundary=boundary_mask_p,
                    numpy_=True).transpose(2, 1, 0)
    
    
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(phi)
    # ax[1].imshow(grad_phi[..., 0])
    # ax[2].imshow(grad_phi[..., 1])
    # ax[0].set_title("phi")
    # ax[1].set_title("grad phi x")
    # ax[2].set_title("grad phi y")
    # plt.show()

    # Compute the new velocity field
    u = u_star - delta_t / density * grad_phi

    # Update pressure
    p = phi + beta * p

    u, p = boundary(geometry=geometry, u=u, p=p)

    global debug_data

    # debug_data.append({"conv_u": np.sum(conv_u),
    #                    "grad_p": np.sum(grad_p),
    #                    "lapl_u": np.sum(lapl_u),
    #                    "div_u_star": np.sum(div_u_star),
    #                    "phi": np.sum(phi),
    #                    "grad_phi": np.sum(grad_phi)})


    div_u = divergence(u=u.transpose(2, 1, 0),
                     dx=dx,
                     dy=dy,
                     boundary=boundary_mask_u,
                     numpy_=True).transpose(1, 0)
    
    div_u = div_u * (1 - boundary_mask)

    # if len(debug_data.keys()) == 0:
    #     debug_data["div_u"] = []
    #     debug_data["conv_u"] = []
    #     debug_data["grad_p"] = []
    #     debug_data["lapl_u"] = []
    #     debug_data["div_u_star"] = []
    #     debug_data["phi"] = []
    #     debug_data["grad_phi"] = []

    # debug_data["div_u"].append(np.sum(div_u ** 2))
    # debug_data["conv_u"].append(np.sum(conv_u ** 2))
    # debug_data["grad_p"].append(np.sum(grad_p ** 2))
    # debug_data["lapl_u"].append(np.sum(lapl_u ** 2))
    # debug_data["div_u_star"].append(np.sum(div_u_star ** 2))
    # debug_data["phi"].append(np.sum(phi ** 2))
    # debug_data["grad_phi"].append(np.sum(grad_phi ** 2))

    # global figdbg
    # global axdbg
    # global keys
    # global lines
    # global img
    # global cbar

    # if figdbg is None:
    #     plt.ion()
    #     # figdbg, axdbg = plt.subplots(2, 4, figsize=(20, 12))
    #     # axdbg = axdbg.flatten()
    #     figdbg, axdbg = plt.subplots(1, 1, figsize=(12, 12))
    #     img = axdbg.imshow(div_u)
    #     cbar = figdbg.colorbar(img, ax=axdbg)
    #     # ax.plot([d["conv_u"] for d in debug_data])

    # #     aaaa = 0
    # #     for key in debug_data.keys():
    # #         if key == "div_u":
    # #             color="red"
    # #         else:
    # #             color="black"
    # #         line, = axdbg[aaaa].plot([], label=key, color=color)
    # #         lines[key] = line
    # #         axdbg[aaaa].set_title(key)
    # #         # axdbg[aaaa].set_yscale("log")
    # #         aaaa += 1

    # #     # plt.legend()

    # img.set_data(div_u)
    # img.set_clim(vmin=np.min(div_u), vmax=np.max(div_u))

    # # for key in debug_data.keys():
    # #     ydata = [d for d in debug_data[key][-10:]]
    # #     xdata = range(len(ydata))
    # #     lines[key].set_data(xdata, ydata)

    # # for aaaa in range(len(axdbg)):
    # #     axdbg[aaaa].relim()
    # #     axdbg[aaaa].autoscale_view()

    # figdbg.canvas.draw()
    # figdbg.canvas.flush_events()
    # plt.pause(0.0001)

        # plt.draw()
        # plt.pause(0.001)
                     
    
    # print(f"{np.mean(div_u)} +- {np.std(div_u)}")

    lap_phi = laplacian(u = phi.transpose(1, 0),
                        dx=dx,
                        dy=dy,
                        boundary=boundary_mask_p,
                        numpy_=True).transpose(1, 0)
    
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(phi)
    # ax[1].imshow(lap_phi)
    # plt.show()
    
    poisson_res = lap_phi - density / delta_t * div_u_star

    dus = div_u_star.ravel()
    dus[p_dirichlet_mask.ravel().astype(bool)] = 0
    res = A_poisson.toarray() @ phi.ravel() - density / delta_t * dus
    # res = res.reshape((8, 8)) * (1 - boundary_mask)
    # plt.imshow(res)
    # plt.show()
    
    # print(f"{np.sum(div_u ** 2)}            {np.sqrt(np.sum(res ** 2))}")

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

    # u, p = boundary(geometry, u, p)

    _, u_dirichlet_mask, u_neumann, _, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, width, height)

    boundary_mask_p =  boundary_mask.transpose(1, 0) * 0

    A_poisson = compute_discrete_laplacian(height, width, dx, dy)

    # i = p_dirichlet_mask.ravel().astype(bool)
    # A_poisson[i, :] = 0
    # A_poisson[i, i] = 1

    # A_poisson = laplacian(np.eye(height * width, height * width), dx=dx, dy=dy, boundary=np.zeros((height * width, height * width)), numpy_=True)
    # A_poisson = sp.csr_matrix(A_poisson)
    # print(A_poisson)

    # test_field = np.random.rand(height, width)
    # test_field[5:20, 10:15] = 1
    # lap_matrix = A_poisson.dot(test_field.ravel()).reshape(height, width)
    # lap_operator = laplacian(test_field.transpose(1, 0), dx=dx, dy=dy, boundary=boundary_mask_p, numpy_=True).transpose(1, 0)

    # vmin = min(np.min(lap_matrix), np.min(lap_operator))
    # vmax = max(np.max(lap_matrix), np.max(lap_operator))

    # fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    # ax[0].imshow(test_field)
    # ax[1].imshow(lap_matrix, vmin=vmin, vmax=vmax)
    # ax[2].imshow(lap_operator, vmin=vmin, vmax=vmax)
    # ax[0].set_title("Test field")
    # ax[1].set_title("Laplacian with matrix")
    # ax[2].set_title("Laplacian with operator")

    # plt.show()
    
    # print(np.max(np.abs(lap_matrix - lap_operator)))

    # exit(0)
    
    
    # A_poisson = A_poisson.toarray()
    # i = 1 - boundary_mask.ravel().astype(bool)
    # A_poisson[i, :] = 0
    # A_poisson[:, i] = 0
    # A_poisson[i, i] = 1
    # A_poisson = sp.csr_matrix(A_poisson)


    if implicit_laplacian:
        A_laplacian = compute_discrete_laplacian(height, width, dx, dy)

        # A_laplacian = A_laplacian.toarray()
        # i = p_dirichlet_mask.ravel().astype(bool)
        # A_laplacian[i, :] = 0
        # A_laplacian[i, i] = 1
        # A_laplacian = sp.csr_matrix(A_laplacian)

        # A_laplacian = compute_A_matrix(height, width, dx, dy)

        # I = sp.eye(width * height)
        # A_laplacian = (I - time_step * viscosity * A_laplacian)
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

        geometry_mtx = geometry_to_mtx(geometry, width, height)

        # Iterate
        for iter in tqdm(range(max_iter), desc="     "):
        
            if method_type == "full_model":

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
                if dynamic_time_step and np.max(np.abs(u)) != 0:
                    time_step = np.min([dx, dy]) * 0.01 / np.max(np.abs(u))

                if A_laplacian == None:
                    A_laplacian_ = None
                else:
                    I = sp.eye(width * height)
                    A_laplacian_ = (I - time_step * viscosity * A_laplacian)

                # Perform one step of the solver
                u_new, p, u_star, div_u_star, phi = navier_stokes_step(u, p, time_step, density, viscosity, beta, dx, dy, geometry, A_poisson, A_laplacian_, model, iter=iter, geometry_mtx=geometry_mtx)

                # if np.max(np.abs(u_new - u)) < 1e-5:
                #     print("\t Velocity converged - stopping early")
                #     break

                # print(np.max(np.abs(u_new - u)))

                u = u_new

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


# def main():
#     u = np.array([[1, 2],
#                  [3, 4]])
    
#     A = compute_discrete_laplacian(2, 2, dx=1, dy=0.1).toarray()

#     print(u)

#     print()

#     print(A)

#     print()

#     print(u.ravel())

# if __name__ == "__main__":
#     main()
