# import numpy as np
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
# import cupyx.scipy.sparse as sp
import scipy.sparse.linalg as spla
# import cupyx.scipy.sparse.linalg as spla
from scipy.sparse.linalg import splu, factorized
# from cupyx.scipy.sparse.linalg import splu, factorized
import torch
import sys
import os
import json
import time
from torch.quantization import quantize_dynamic
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import matplotlib.pyplot as plt
import torch.nn.functional as F

import gc

sys.path.insert(0, "..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.model import CNN
from modeling.model_encdec import EncDecCNN
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
    # A = sp.csr_matrix((n * m, n * m))

    # Fill diagonal blocks
    for i in range(n):
        A[i * m:(i + 1) * m, i * m:(i + 1) * m] = D

    # x-direction neighbors
    A += dx_inv * sp.eye(n * m, k=m) + dx_inv * sp.eye(n * m, k=-m)

    A = sp.csr_matrix(A)
    return A


def solve_discrete_poisson(u, A):
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
    # phi_flat, info = spla.cg(A, u)
    phi_flat = spla.spsolve(A, u)
    # print("CG info:", info)

    # Reshape the solution vector into a matrix
    phi = phi_flat.reshape((n, m))

    return phi

def navier_stokes_step(u, p, delta_t, density, viscosity, beta, dx, dy, geometry, A_poisson, A_laplacian = None, poisson_model = None, iter = 0, geometry_mtx=None, x_mean=None, x_std=None, poisson_alternation=-1):
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

    u_bc, p_bc = boundary(geometry=geometry, u=u, p=p)

    # fig, ax = plt.subplots(5, 5)
    # ax = ax.flatten()
    # ax[0].imshow(u[..., 0])
    # ax[0].set_title("u[0]")

    # ax[1].imshow(u[..., 1])
    # ax[1].set_title("u[1]")

    # ax[2].imshow(p)
    # ax[2].set_title("p")

    # ax[3].imshow(u_bc[..., 0])
    # ax[3].set_title("u_bc[0]")

    # ax[4].imshow(u_bc[..., 1])
    # ax[4].set_title("u_bc[1]")

    # ax[5].imshow(p_bc)
    # ax[5].set_title("p_bc")

    # fig, ax = plt.subplots(1, 2)
    # im0 = ax[0].imshow(u_bc[..., 0])
    # im1 = ax[1].imshow(u_bc[..., 1])

    # ax[0].set_title("x")
    # ax[1].set_title("y")

    # plt.colorbar(im0)
    # plt.colorbar(im1)
    # plt.show()
    # exit(0)

    # Handle the Laplacian term implicitly
    if A_laplacian is not None:

        u_star = u

        # Extract shape
        n, m, _ = u_star.shape

        # Compute the convective term
        conv_u = convective(u = u_bc,
                          dx = dx,
                          dy = dy,
                          boundary=None,
                          numpy_=True)
        
        # fig, ax = plt.subplots(1, 2)
        # im0 = ax[0].imshow(conv_u[..., 0])
        # im1 = ax[1].imshow(conv_u[..., 1])

        # ax[0].set_title("x")
        # ax[1].set_title("y")

        # plt.colorbar(im0)
        # plt.colorbar(im1)
        # plt.show()
        # exit(0)

        # Compute the pressure gradient
        grad_p = grad(u=p_bc,
                      dx=dx,
                      dy=dy,
                      boundary=None,
                      numpy_=True)
        
        # full_lapl = laplacian(u=u_bc, dx=dx, dy=dy, numpy_=True)
    
        # u_only_interior = np.pad(u, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        # interior_lapl = laplacian(u=u_only_interior, dx=dx, dy=dy, numpy_=True)

        # boundary_injection = full_lapl - interior_lapl

        # Compute right hand side for implicit Laplacian system
        # TODO: Find a source for this last term
        b = (u - delta_t * conv_u - beta * delta_t / density * grad_p) # + (delta_t * viscosity * boundary_injection)

        # fig, ax = plt.subplots(1, 2)
        # im0 = ax[0].imshow(b[..., 0])
        # im1 = ax[1].imshow(b[..., 1])

        # ax[0].set_title("x")
        # ax[1].set_title("y")

        # plt.colorbar(im0)
        # plt.colorbar(im1)
        # plt.show()
        # exit(0)

        # Reshape
        b_x = b[:, :, 0].ravel()
        b_y = b[:, :, 1].ravel()

        # Solve the system
        # u_star_x = spla.spsolve(A_laplacian, b_x)
        # u_star_y = spla.spsolve(A_laplacian, b_y)

        # lu = splu(A_laplacian)
        u_star_x = A_laplacian.solve(b_x)
        u_star_y = A_laplacian.solve(b_y)

        # Reshape the solution
        u_star = np.zeros((n, m, 2))
        u_star[:, :, 0] = u_star_x.reshape((n, m))
        u_star[:, :, 1] = u_star_y.reshape((n, m))
    else:
        # Compute convective term
        conv_u = convective(u=u_bc,
                          dx=dx,
                          dy=dy,
                          boundary=None,
                          numpy_=True)


        # Compute pressure gradient        
        grad_p = grad(u=p_bc,
                      dx=dx,
                      dy=dy,
                      boundary=None,
                      numpy_=True)
        

        # Compute laplacian term        
        lapl_u = laplacian(u=u_bc,
                           dx=dx,
                           dy=dy,
                           boundary=None,
                           numpy_=True)

        # Compute intermediate velocity
        u_star = u_bc - delta_t * conv_u - beta * delta_t / density * grad_p + delta_t * viscosity * lapl_u

        # ax[6].imshow(conv_u[..., 0])
        # ax[6].set_title("conv_u[0]")

        # ax[7].imshow(conv_u[..., 1])
        # ax[7].set_title("conv_u[1]")

        # ax[8].imshow(grad_p[..., 0])
        # ax[8].set_title("grad_p[0]")

        # ax[9].imshow(grad_p[..., 1])
        # ax[9].set_title("grad_p[1]")

        # ax[10].imshow(lapl_u[..., 0])
        # ax[10].set_title("lapl_u[0]")

        # ax[11].imshow(lapl_u[..., 1])
        # ax[11].set_title("lapl_u[1]")

        # ax[12].imshow(u_star[..., 0])
        # ax[12].set_title("u_star[0]")

        # ax[13].imshow(u_star[..., 1])
        # ax[13].set_title("u_star[1]")


    # Apply boundary
    u_star_bc, _ = boundary(geometry, u = u_star)

    # ax[14].imshow(u_star_bc[..., 0])
    # ax[14].set_title("u_star_bc[0]")

    # ax[15].imshow(u_star_bc[..., 1])
    # ax[15].set_title("u_star_bc[1]")

    # Compute divergence of u star
    div = divergence(u=u_star_bc,
                     dx=dx,
                     dy=dy,
                     boundary=None,
                     numpy_=True)
    
    # plt.imshow(div)
    # plt.colorbar()
    # plt.show()
    # exit(0)

    # div_before = div.mean()

    # Scale the divergence of u star
    div_u_star = (density / delta_t) * div

    # print(density, delta_t, density/delta_t)

    # fig, ax = plt.subplots(1, 3)
    # im0 = ax[0].imshow(u_star[..., 0])
    # im1 = ax[1].imshow(u_star[..., 1])
    # im2 = ax[2].imshow(div_u_star)

    # plt.colorbar(im0)
    # plt.colorbar(im1)
    # plt.colorbar(im2)

    # plt.show()

    # ax[16].imshow(div_u_star)
    # ax[16].set_title("div_u_star")


    # plt.imshow(bmi)
    # plt.show()

    # _, u_dirichlet_mask, u_neumann, _, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, n, m)

    # Solve Poisson's equation using a model
    if poisson_model is not None and (poisson_alternation == -1 or iter % poisson_alternation != 0): #and iter % 5 != 0:

        with torch.no_grad():
            n, m = div_u_star.shape
            device = torch.device('cpu')

            # Convert u star to Torch
            u_star_ = torch.from_numpy(u_star).float().to(device).permute(2, 0, 1)
            div_u_star_ = torch.from_numpy(div_u_star).float().to(device).unsqueeze(0)

            # Create a coordinate grid
            grid_y, grid_x = torch.meshgrid(
                    torch.linspace(0, 1, m),
                    torch.linspace(0, 1, n),
                    indexing="xy"
                )
            coords = torch.stack([grid_x, grid_y], dim=0)

            # Obtain required boundary representations
            u_dirichlet, u_dirichlet_mask, u_neumann, p_dirichlet, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, m, n)

            # Convert boundaries to Torch
            p_neumann = torch.tensor(p_neumann, dtype=torch.float32).permute(2, 0, 1)
            p_dirichlet = torch.tensor(p_dirichlet, dtype=torch.float32).unsqueeze(0)
            p_dirichlet_mask = torch.tensor(p_dirichlet_mask, dtype=torch.float32).unsqueeze(0)

            rho = torch.ones((1, n, m)) * density
            delta_t_ = torch.ones((1, n, m)) * delta_t
            dx_ = torch.ones((1, n, m)) * dx
            dy_ = torch.ones((1, n, m)) * dy
            boundary_mask_ = torch.tensor(boundary_mask, dtype=torch.float32).unsqueeze(0)

            # Construct the model's input tensor
            # u_star_coords = torch.cat([u_star_, 
            #                            coords, 
            #                            p_neumann, 
            #                            p_dirichlet,
            #                            p_dirichlet_mask,
            #                            rho,
            #                            delta_t_,
            #                            boundary_mask_
            #                         ], dim=0).unsqueeze(0)
                                    

            u_star_coords = torch.cat([0 * div_u_star_, # TODO: THIS IS ZERO
                                       coords, 
                                       p_dirichlet,
                                       p_dirichlet_mask,
                                       boundary_mask_,
                                       u_star_,
                                       p_neumann, 
                                       rho,
                                       delta_t_,
                                       dx_,
                                       dy_
                                    ], dim=0).unsqueeze(0)

            # fig, ax = plt.subplots(3, 4)
            # ax = ax.flatten()
            # for j in range(0, 12):
            #     im = ax[j].imshow(u_star_coords[-1][j].cpu().detach().numpy())
            #     plt.colorbar(im)

            # plt.show()

            # print(u_star_.max())

            u_star_coords = (u_star_coords - x_mean) / x_std

            # Perform a forward pass and obtain the pressure correction
            phi = poisson_model(u_star_coords)
            phi = F.pad(phi, (1, 1, 1, 1), mode="constant", value=0)
            phi = phi.view(n, m).cpu().numpy()
    else:
        # Obtain the pressure correction by numerically solving Poisson's equatio
        phi = solve_discrete_poisson(div_u_star, A=A_poisson)
        # phi = solve_discrete_poisson_jacobi(div_u_star, dx, dy, geometry)

    # ax[17].imshow(phi)
    # ax[17].set_title("phi")

    # Apply the boundary conditions to the pressure correction
    _, phi_bc = boundary(geometry, p = phi)

    # ax[18].imshow(phi_bc)
    # ax[18].set_title("phi_bc")

    # Compute the gradient of the pressure correction
    grad_phi = grad(u=phi_bc,
                    dx=dx,
                    dy=dy,
                    boundary=None,
                    numpy_=True)
    
    # ax[19].imshow(grad_phi[..., 0])
    # ax[19].set_title("grad_phi[0]")

    # ax[20].imshow(grad_phi[..., 1])
    # ax[20].set_title("grad_phi[1]")

    # plt.show()

    # Compute the new velocity field
    u = u_star_bc - delta_t / density * grad_phi

    # div = divergence(u, dx=dx, dy=dy, boundary=None, numpy_=True)

    # div = div ** 2

    # _, u_dirichlet_mask, u_neumann, _, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, n, m)
    # div *= (1 - boundary_mask)

    # print(f"{div.mean():.5f} +- {div.std():.5f}")

    # Update pressure
    p = phi_bc + beta * p

    u, p = boundary(geometry=geometry, u=u, p=p)

    return u, p, u_star_bc, div_u_star, phi_bc

def solve_navier_stokes(width, height, dx, dy, max_iter, time_step, density, viscosity, beta, geometry, dynamic_time_step = False, implicit_laplacian = False, return_div_u_star = False, method_type = "numeric", model_path = None, poisson_alternation=-1):
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

    # Shrink the domain by one point on each side to allow for padding with boundary values
    # height = height - 2
    # width = width - 2

    # Initialize velocity and pressure
    u = np.zeros((height, width, 2))
    p = np.zeros((height, width))

    # u, p = boundary(geometry, u, p)

    _, u_dirichlet_mask, u_neumann, _, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, width, height)

    A_poisson = compute_discrete_laplacian(height, width, dy, dx)
    A_laplacian_init = compute_discrete_laplacian(height, width, dy, dx)

    # randu = np.random.rand(height, width)

    # randu[5:12, 3:30] = 1
    # randu[33:45, 45:55] = 0

    # lap_mtx = A_poisson @ randu.ravel()
    # lap_op = laplacian(randu, dx, dy, None, numpy_=True)

    # plt.ioff()
    # fig, ax = plt.subplots(figsize=(16, 9), nrows=1, ncols=4)
    # ax[0].imshow(randu)
    # ax[1].imshow(lap_mtx.reshape((height, width)))
    # ax[2].imshow(lap_op)
    # ax[3].imshow((lap_mtx.reshape((height, width)) - lap_op) ** 2)
    # plt.show()

    time_step_arg = time_step

    # if implicit_laplacian:
    #     # A_laplacian = compute_discrete_laplacian(height, width, dx, dy)
    #     A_laplacian = A_poisson
    #     I = sp.eye(width * height)
    #     A_laplacian = (I - time_step * viscosity * A_laplacian)
    #     A_laplacian = splu(A_laplacian)
    # else:
    #     A_laplacian = None

    if method_type in ["full_model", "poisson_model"]:
        with open(os.path.join(model_path, "params.json"), "r") as f:
            params = json.load(f)

        x_mean = torch.tensor(params["x_mean"]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x_std = torch.tensor(params["x_std"]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        with torch.no_grad():
            if method_type == "full_model":
                dataset = FullPINNDataset(data_path = None, N = 0, device="cpu")
            elif method_type == "poisson_model":
                dataset = PoissonPINNDataset(data_path = None, N = 0, device="cpu")
                
            input_channels, _, _, output_channels = dataset.get_dims()

            if params["architecture"] == "cnn": #or method_type == "poisson_model": # TODO: Remove this last case
                model = CNN(input_channels=input_channels,
                            output_channels=output_channels,
                            num_layers=int(params["num_layers"]),
                            hidden_channels=int(params["hidden_dim"]),
                            input_width=width,
                            input_height=height,
                            enforceBC=dataset.enforceBC,
                            use_skip=True,
                            activation=params["activation"],
                            architecture=params["architecture"],
                            pooling=params["pooling"],
                            upsampling=params["upsampling"],
                            norm=params["norm"],
                        )
            elif params["architecture"] == "encdec":
                model = EncDecCNN(input_channels=input_channels,
                                output_channels=output_channels,
                                num_layers=int(params["num_layers"]),
                                hidden_channels=int(params["hidden_dim"]),
                                input_width=width,
                                input_height=height,
                                enforceBC=dataset.enforceBC,
                                use_skip=True,
                                activation=params["activation"],
                                architecture=params["architecture"],
                                pooling=params["pooling"],
                                upsampling=params["upsampling"],
                                norm=params["norm"],
                            )

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
        x_mean = None
        x_std = None

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

        grid_y, grid_x = torch.meshgrid(
                    torch.linspace(0, 1, width),
                    torch.linspace(0, 1, height),
                    indexing="xy"
                )
        coords = torch.stack([grid_x, grid_y], dim=0)

        rho = torch.ones((1, height, width)) * density
        mu = torch.ones((1, height, width)) * viscosity
        dx_ = torch.ones((1, height, width)) * dx
        dy_ = torch.ones((1, height, width)) * dy
        boundary_mask_ = torch.tensor(boundary_mask, dtype=torch.float32).unsqueeze(0)
        
        # print(coords.shape)
        # print(u_dirichlet.permute(2, 0, 1).shape)
        # print(u_dirichlet_mask.permute(2, 0, 1).shape)
        # print(u_neumann.permute(2, 0, 1).shape)
        # print(p_dirichlet.unsqueeze(0).shape)
        # print(p_dirichlet_mask.unsqueeze(0).shape)
        # print(p_neumann.permute(2, 0, 1).shape)
        # print(rho.shape)
        # print(mu.shape)
        # print(boundary_mask_.shape)

        x = torch.cat([coords,
                       u_dirichlet.permute(2, 0, 1),
                       u_dirichlet_mask.permute(2, 0, 1),
                       u_neumann.permute(2, 0, 1),
                       p_dirichlet.unsqueeze(0),
                       p_dirichlet_mask.unsqueeze(0),
                       p_neumann.permute(2, 0, 1),
                       rho,
                       mu,
                       boundary_mask_,
                       dx_,
                       dy_
                    ], dim=0).unsqueeze(0)
        
        x = (x - x_mean) / x_std

        out = model(x)#.squeeze(0).detach().numpy()

        out = dataset.enforceBC(x, out).squeeze(0).detach().numpy()

        u_x, u_y, p = out[0], out[1], out[2]
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

        # u_dirichlet, u_dirichlet_mask, u_neumann, p_dirichlet, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, width, height)

        # u_dirichlet = torch.tensor(u_dirichlet, dtype=torch.float32)
        # u_dirichlet_mask = torch.tensor(u_dirichlet_mask, dtype=torch.float32)
        # u_neumann = torch.tensor(u_neumann, dtype=torch.float32)
        # p_dirichlet = torch.tensor(p_dirichlet, dtype=torch.float32)
        # p_dirichlet_mask = torch.tensor(p_dirichlet_mask, dtype=torch.float32)
        # p_neumann = torch.tensor(p_neumann, dtype=torch.float32)

        # grid_x, grid_y = torch.meshgrid(
        #         torch.linspace(0, width * dx, width),
        #         torch.linspace(0, height * dy, height),
        #         indexing="ij"
        #     )
        
        # coords = torch.stack([grid_y, grid_x], dim=0)

        geometry_mtx = geometry_to_mtx(geometry, width, height)

        # Iterate
        for iter in tqdm(range(max_iter), desc="     "):
        
            # if dynamic_time_step and np.max(np.abs(u)) != 0:
            #     time_step = np.min([dx, dy]) * time_step_arg / np.max(np.abs(u))

            if dynamic_time_step:
                u_max = np.max(np.abs(u[..., 0]))
                v_max = np.max(np.abs(u[..., 1]))

                dt_adv = min(dx / max(u_max, 1e-6), dy / max(v_max, 1e-6))

                dt_diff = 0.5 / (viscosity * (1 / dx ** 2 + 1 / dy ** 2))

                CFL = time_step_arg
                dt_new = CFL * min(dt_adv, dt_diff)

                time_step = np.clip(dt_new, 1e-6, np.inf)


            # if A_laplacian is None:
            #     A_laplacian_ = None
            # else:
            #     I = sp.eye(width * height)
            #     A_laplacian_ = (I - time_step * viscosity * A_laplacian)

            if implicit_laplacian:
                # A_laplacian = compute_discrete_laplacian(height, width, dx, dy)
                A_laplacian = A_laplacian_init
                I = sp.eye(width * height)
                A_laplacian = (I - time_step * viscosity * A_laplacian)

                A_laplacian = splu(A_laplacian)
            else:
                A_laplacian = None

            # Perform one step of the solver
            u_new, p_new, u_star, div_u_star, phi = navier_stokes_step(u, p, time_step, density, viscosity, beta, dx, dy, geometry, A_poisson, A_laplacian, model, iter=iter, geometry_mtx=geometry_mtx, x_mean=x_mean, x_std=x_std, poisson_alternation=poisson_alternation)

            # if np.max(np.abs(u - u_new)) < 1e-5 and np.max(np.abs(p - p_new)) < 1e-5:
            #     print(f"\t Converged after {iter} iterations.")
            #     break

            u = u_new
            p = p_new

            if np.isnan(np.min(u)) or np.isnan(np.min(p)):
                print("\t NaN found - stopping early")
                break
            
            # Save results
            results_u.append(u)
            results_p.append(p)
            results_u_star.append(u_star)
            results_div_u_star.append(div_u_star)
            results_phi.append(phi)

            # if iter % 10 == 0:
            #     gc.collect()

    if return_div_u_star:
        return results_u, results_p, results_u_star, results_div_u_star, results_phi
    else:
        return results_u, results_p