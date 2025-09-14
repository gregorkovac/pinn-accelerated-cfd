import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import json

def load_case(data_path, idx, samples_per_case=-1):
    case_path = os.path.join(data_path, f"{idx}")

    u_star = np.load(os.path.join(case_path, "u_star.npy"))
    phi = np.load(os.path.join(case_path, "phi.npy"))

    with open(os.path.join(case_path, "params.json")) as f:
        params = json.load(f)

    if samples_per_case == -1:
        samples_per_case = len(u_star)

    indices = np.random.choice(samples_per_case, samples_per_case, replace=False)

    return u_star[indices], phi[indices], params["dx"], params["dy"], params["time_step"], params["density"], params["viscosity"], params["geometry"] #, params["cylinder_x"], params["cylinder_y"], params["cylinder_r"]

def geometry_to_mtx(geometry, width, height):
    # TODO: Implement BC-s for velocity - when dealing with full model

    # Initialize tensor for storing normals for boundary conditions on pressure and velocity
    p_dirichlet = np.zeros((height, width))
    p_dirichlet_mask = np.zeros((height, width))
    p_neumann = np.zeros((height, width, 2))

    u_dirichlet = np.zeros((height, width, 2))
    u_dirichlet_mask = np.zeros((height, width))
    u_neumann = np.zeros((height, width, 2))

    # Initialize a binary mask for storing at which points the boundary is present
    boundary_mask = np.zeros((height, width))

    cylinder_mask = None

    for g in geometry:

        # For walls and inflows, set a Neumann boundary condition for pressure
        if g["type"] in ["wall", "inflow"]:
            
            # Extract the region
            x, y = g["pos"]["x"], g["pos"]["y"]
            w, h = g["size"]["width"], g["size"]["height"]

            # The normal is defined either by a direction or the velocity of the inflow
            if "dir" in g.keys():
                xn, yn = g["dir"]["x"], g["dir"]["y"]
            elif "vel" in g.keys():
                xn, yn = g["vel"]["x"], g["vel"]["y"]
            else:
                continue

            if xn != 0:
                xn = np.sign(xn)
            if yn != 0:
                yn = np.sign(yn)

            # Convert the "-1" notation to width/height
            if x == -1:
                x = width - 1
            if y == -1:
                y = height - 1
            if w == -1:
                # TODO: ?
                w = width
            if h == -1:
                h = height + 1

            # Add values to the pressure Neuman tensor
            # TODO: For inflow normalize this to (0, 1). For example, we can have (0, 5) as inflow

            # Add values to the boundary mask
            boundary_mask[y:(y+h), x:(x+w)] = 1
            p_neumann[y:(y+h), x:(x+w)] = np.array([yn, xn])

            if g["type"] == "inflow":
                u_dirichlet[y:(y+h), x:(x+w)] = np.array([yn, xn])
                u_dirichlet_mask[y:(y+h), x:(x+w)] = 1
            # else:
            if g["type"] == "wall":
                u_dirichlet[y:(y+h), x:(x+w)] = np.array([0, 0])
                u_dirichlet_mask[y:(y+h), x:(x+w)] = 1

        elif g["type"] == "outflow":
            # Extract the region
            x, y = g["pos"]["x"], g["pos"]["y"]
            w, h = g["size"]["width"], g["size"]["height"]
            xn, yn = g["dir"]["x"], g["dir"]["y"]

            # Convert the "-1" notation to width/height
            if x == -1:
                x = width - 1
            if y == -1:
                y = height - 1
            if w == -1:
                # TODO: ?
                w = width
            if h == -1:
                h = height + 1

            boundary_mask[y:(y+h), x:(x+w)] = 1

            p_dirichlet[y:(y+h), x:(x+w)] = 0
            p_dirichlet_mask[y:(y+h), x:(x+w)] = 1

            # u_neumann[y:(y+h), x:(x+w)] = np.array([yn, xn])
        elif g["type"] == "cylinder":
            # Extract coordinates and radius
            x, y, r = g["pos"]["x"], g["pos"]["y"], g["size"]

            # Determine min and max values in each coordinate direction
            x_min = max(0, x - r)
            x_max = min(width, x + r + 1)
            y_min = max(0, y - r)
            y_max = min(height, y + r + 1)

            # Define a grid
            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]

            # Distance check
            dist_sq = (xx - x) ** 2 + (yy - y) ** 2

            # Mask out points inside the circle
            mask = dist_sq <= r ** 2

            # Set the values in the boundary mask
            boundary_mask[y_min:y_max, x_min:x_max][mask] = 1

            u_dirichlet[y_min:y_max, x_min:x_max][mask] = np.array([0, 0])
            u_dirichlet_mask[y_min:y_max, x_min:x_max][mask] = 1

            # TODO: Check this code
            # dx = xx - x
            # dy = yy - y
            # norm = np.sqrt(dx**2 + dy**2) + 1e-12
            # nx = dx / norm
            # ny = dy / norm
            # p_neumann[y_min:y_max, x_min:x_max][mask] = np.stack([ny[mask], nx[mask]], axis=-1)

            cylinder_mask = np.zeros((height, width), dtype=bool)
            cylinder_mask[y_min:y_max, x_min:x_max][mask] = 1

    # Remove entries where the vector is pointing to another boundary point
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    ny = yy + p_neumann[..., 0].astype(int)
    nx = xx + p_neumann[..., 1].astype(int)
    invalid = (
        (boundary_mask == 1) &
        ((ny < 0) | (ny >= height) | (nx < 0) | (nx >= width) |
         (boundary_mask[np.clip(ny, 0, height-1), np.clip(nx, 0, width-1)] == 1))
    )
    if cylinder_mask is not None:
        invalid[cylinder_mask] = False
    p_neumann[invalid] = (0, 0)

    # Remove entries where the vector is pointing to another boundary point
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    ny = yy + u_neumann[..., 0].astype(int)
    nx = xx + u_neumann[..., 1].astype(int)
    invalid = (
        (boundary_mask == 1) &
        ((ny < 0) | (ny >= height) | (nx < 0) | (nx >= width) |
         (boundary_mask[np.clip(ny, 0, height-1), np.clip(nx, 0, width-1)] == 1))
    )
    u_neumann[invalid] = (0, 0)

    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Rectangle

    # x = np.arange(width)
    # y = np.arange(height)
    # X, Y = np.meshgrid(x, y)   
    
    # U = p_neumann[..., 1]                      # (n, m)
    # V = p_neumann[..., 0]                      # (n, m)
    
    # fig, ax = plt.subplots(figsize=(8, 8))
    # for i in range(height):
    #     for j in range(width):
    #         rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
    #                      edgecolor='gray', facecolor='none', linewidth=0.5)
    #         ax.add_patch(rect)
    # # ax.scatter(X, Y, color='black', s=10, alpha=0.8, label='Grid points')
    # q = ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, pivot='mid')
    # ax.set_aspect('equal')

    # plt.show()

    return u_dirichlet, u_dirichlet_mask, u_neumann, p_dirichlet, p_dirichlet_mask, p_neumann, boundary_mask

class PINNDataset(Dataset):
    def __init__(self, data_path, N, samples_per_case=-1):
        if data_path == None:
            self.height = 0
            self.width = 0
            return

        # Initialize variables
        self.data_len = 0
        self.u_star = []
        self.phi = []
        self.delta_x = []
        self.delta_y = []
        self.delta_t = []
        self.t = []
        self.rho = []
        self.mu = []
        self.u_dirichlet = []
        self.u_dirichlet_mask = []
        self.u_neumann = []
        self.p_dirichlet = []
        self.p_dirichlet_mask = []
        self.p_neumann = []
        self.boundary_mask = []

        # Iterate over cases
        for i in range(N):
            u_star, phi, dx, dy, time_step, rho, mu, geometry = load_case(data_path, i, samples_per_case)

            # Save the case shape
            if i == 0:
                shape = np.shape(u_star)
                self.height, self.width = shape[1], shape[2]

                # Create a coordinate grid
                grid_x, grid_y = torch.meshgrid(
                        torch.linspace(0, self.width * dx, self.width),
                        torch.linspace(0, self.height * dy, self.height),
                        indexing="ij"
                )
                self.coords = torch.stack([grid_y, grid_x], dim=0)

            u_dirichlet, u_dirichlet_mask, u_neumann, p_dirichlet, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, self.width, self.height)

            # Iterate over time steps
            for j in range(len(u_star)):
                self.u_star.append(u_star[j])
                self.phi.append(phi[j])
                self.delta_x.append(dx)
                self.delta_y.append(dy)
                self.rho.append(rho)
                self.mu.append(mu)

                t = torch.ones(self.height, self.width) * j * time_step
                t = t.unsqueeze(0)

                self.t.append(t)
                self.delta_t.append(time_step)

                self.u_dirichlet.append(u_dirichlet)
                self.u_dirichlet_mask.append(u_dirichlet_mask)
                self.u_neumann.append(u_neumann)
                self.p_dirichlet.append(p_dirichlet)
                self.p_dirichlet_mask.append(p_dirichlet_mask)
                self.p_neumann.append(p_neumann)
                self.boundary_mask.append(boundary_mask)

            self.data_len += len(u_star)


        # Convert to torch tensors
        self.u_star = torch.tensor(np.array(self.u_star), dtype=torch.float32)
        self.phi = torch.tensor(np.array(self.phi), dtype=torch.float32)
        self.delta_x = torch.tensor(np.array(self.delta_x), dtype=torch.float32)
        self.delta_y = torch.tensor(np.array(self.delta_y), dtype=torch.float32)
        self.delta_t = torch.tensor(np.array(self.delta_t), dtype=torch.float32)
        self.t = torch.tensor(np.array(self.t), dtype=torch.float32)
        self.rho = torch.tensor(np.array(self.rho), dtype=torch.float32)
        self.mu = torch.tensor(np.array(self.mu), dtype=torch.float32)
        # self.boundary = torch.tensor(np.array(self.boundary), dtype=torch.float32)
        self.u_dirichlet = torch.tensor(np.array(self.u_dirichlet), dtype=torch.float32)
        self.u_dirichlet_mask = torch.tensor(np.array(self.u_dirichlet_mask), dtype=torch.float32)
        self.u_neumann = torch.tensor(np.array(self.u_neumann), dtype=torch.float32)
        self.p_dirichlet = torch.tensor(np.array(self.p_dirichlet), dtype=torch.float32)
        self.p_dirichlet_mask = torch.tensor(np.array(self.p_dirichlet_mask), dtype=torch.float32)
        self.p_neumann = torch.tensor(np.array(self.p_neumann), dtype=torch.float32)
        self.boundary_mask = torch.tensor(np.array(self.boundary_mask), dtype=torch.float32)

        mean = self.u_star[..., :2].mean(dim=(0, 1, 2)) 
        std = self.u_star[..., :2].std(dim=(0, 1, 2))
        # self.u_star[..., 0] = (self.u_star[..., 0] - mean[0]) / (std[0] + 1e-6)
        # self.u_star[..., 1] = (self.u_star[..., 1] - mean[1]) / (std[1] + 1e-6)

        self.u_star_mean = mean
        self.u_star_std = std

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return NotImplementedError
    
    def get_dims(self):
        return self.input_channels, self.height, self.width, self.output_channels
    
    def enforceBC(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return NotImplementedError
    
class PoissonPINNDataset(PINNDataset):
    def __init__(self, data_path, N, samples_per_case=-1):
        super().__init__(data_path, N, samples_per_case)
        self.input_channels = 8
        self.output_channels = 1

    def __getitem__(self, idx):

        x = torch.cat([self.u_star[idx].permute(2, 0, 1), # 0, 1
                       self.coords, # 2, 3
                       self.p_neumann[idx].permute(2, 0, 1), # 4, 5
                       self.p_dirichlet[idx].unsqueeze(0), # 6
                       self.p_dirichlet_mask[idx].unsqueeze(0)], dim=0) # 7

        return x, self.delta_x[idx], self.delta_y[idx], self.t[idx], self.delta_t[idx], self.rho[idx], self.mu[idx], self.coords, self.boundary_mask[idx]
    
    def get_phi(self, idx):
        return self.phi[idx]
    
    def enforceBC(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        p_dirichlet = x[:, 6:7]
        p_dirichlet_mask = x[:, 7:8]

        return out * (1 - p_dirichlet_mask) + p_dirichlet * p_dirichlet_mask

class FullPINNDataset(PINNDataset):
    def __init__(self, data_path, N, samples_per_case=-1):
        super().__init__(data_path, N, samples_per_case)
        self.input_channels = 11
        self.output_channels = 3

    def __getitem__(self, idx):

        x = torch.cat([self.coords, # 0, 1
                       self.u_dirichlet[idx].permute(2, 0, 1), # 2, 3
                       self.u_dirichlet_mask[idx].unsqueeze(0), # 4
                       self.u_neumann[idx].permute(2, 0, 1), # 5, 6
                       self.p_dirichlet[idx].unsqueeze(0), # 7
                       self.p_dirichlet_mask[idx].unsqueeze(0), # 8
                       self.p_neumann[idx].permute(2, 0, 1), # 9, 10
                    ], dim=0) 

        return x, self.delta_x[idx], self.delta_y[idx], self.t[idx], self.delta_t[idx], self.rho[idx], self.mu[idx], self.coords, self.boundary_mask[idx]
    
    def enforceBC(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        u_dirichlet = x[:, 2:4]
        u_dirichlet_mask = x[:, 4:5].expand(-1, 2, -1, -1)
        p_dirichlet = x[:, 7:8]
        p_dirichlet_mask = x[:, 8:9]

        u = out[:, :2]
        p = out[:, 2:3]

        u = u * (1 - u_dirichlet_mask) + u_dirichlet * u_dirichlet_mask
        p = p * (1 - p_dirichlet_mask) + p_dirichlet * p_dirichlet_mask

        return torch.cat([u, p], dim=1)

