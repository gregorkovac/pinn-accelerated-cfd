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

    # u_star = u_star[(len(u_star) // 2):]
    # phi = phi[(len(phi) // 2):]

    with open(os.path.join(case_path, "params.json")) as f:
        params = json.load(f)

    if samples_per_case == -1:
        samples_per_case = len(u_star)

    indices = np.random.choice(samples_per_case, samples_per_case, replace=False)

    return u_star[indices], phi[indices], params["dx"], params["dy"], params["time_step"], params["density"], params["viscosity"], params["geometry"] #, params["cylinder_x"], params["cylinder_y"], params["cylinder_r"]

def geometry_to_mtx(geometry, width, height):
    # print(geometry)
    # TODO: Implement BC-s for velocity - when dealing with full model

    # p_dirichlet = np.zeros((height, width))
    # p_dirichlet_vals = np.zeros((height, width))
    p_neumann = np.zeros((height, width, 2))
    # test = np.zeros((height, width))

    boundary_mask = np.zeros((height, width))

    for g in geometry:
        # print(g)

        if g["type"] in ["wall", "inflow"]:
            x, y = g["pos"]["x"], g["pos"]["y"]
            w, h = g["size"]["width"], g["size"]["height"]

            if x == -1:
                x = width - 1
            if y == -1:
                y = height - 1
            if w == -1:
                w = width
            if h == -1:
                h = height + 1

            w = w - 1
            h = h - 1

            if x - 1 > 0 and ((g["type"] == "inflow" and g["vel"]["x"] < 0) or g["type"] == "wall"):
                p_neumann[y:(y+h+1), x] = np.array([0, -1])
                boundary_mask[y:(y+h+1), x] = 1

            if x + w + 1 < width and ((g["type"] == "inflow" and g["vel"]["x"] > 0) or g["type"] == "wall"):
                p_neumann[y:(y+h+1), x+w] = np.array([0, 1])
                boundary_mask[y:(y+h+1), x+w] = 1
            
            if y - 1 > 0 and ((g["type"] == "inflow" and g["vel"]["y"] < 0) or g["type"] == "wall"):
                p_neumann[y, x:(x+w+1)] = np.array([-1, 0])
                boundary_mask[y, x:(x+w+1)] = 1

            if y + h + 1 < height and ((g["type"] == "inflow" and g["vel"]["y"] > 0) or g["type"] == "wall"):
                p_neumann[y+h, x:(x+w+1)] = np.array([1, 0])
                boundary_mask[y+h, x:(x+w+1)] = 1
        
        elif g["type"] in ["inflow"]:
            x, y = g["pos"]["x"], g["pos"]["y"]
            w, h = g["size"]["width"], g["size"]["height"]

            if x == -1:
                x = width - 1
            if y == -1:
                y = height - 1
            if w == -1:
                w = width
            if h == -1:
                h = height + 1

            # w = w - 1
            # h = h - 1

            boundary_mask[y:(y+h), x:(x+w)] = 1

        elif g["type"] == "cylinder":
            x, y, r = g["pos"]["x"], g["pos"]["y"], g["size"]

            x_min = max(0, x - r)
            x_max = min(width, x + r + 1)
            y_min = max(0, y - r)
            y_max = min(height, y + r + 1)

            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]

            dist_sq = (xx - x) ** 2 + (yy - y) ** 2

            mask = dist_sq <= r ** 2

            boundary_mask[y_min:y_max, x_min:x_max][mask] = 1


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

    # exit(0)

    return p_neumann, boundary_mask

class PINNDataset(Dataset):
    def __init__(self, data_path, N, samples_per_case=-1):
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
        self.boundary = []
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

            boundary, boundary_mask = geometry_to_mtx(geometry, self.width, self.height)

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

                # grid_x = coords[:, 0, :, :]  # shape: (B, H, W)
                # grid_y = coords[:, 1, :, :]  # shape: (B, H, W)

                # # Extract centers and radii: (B, 1, 1) for broadcasting
                # cx = boundary[:, 0].view(-1, 1, 1)
                # cy = boundary[:, 1].view(-1, 1, 1)
                # r  = boundary[:, 2].view(-1, 1, 1)

                # # Mask of cylinder interior: (B, H, W)
                # mask = ((grid_x - cx) ** 2 + (grid_y - cy) ** 2) <= (r ** 2)

                # # Velocity inside cylinder
                # u_inside = x[:, 0] * mask
                # v_inside = x[:, 1] * mask

                # loss_bc_cylinder = torch.mean(u_inside ** 2) + torch.mean(v_inside ** 2)

                # mask = ((grid_x / dx - cylinder_x) ** 2 + (grid_y / dy - cylinder_y) ** 2) <= (cylinder_r ** 2)

                # mask = mask * 0


                # mask[:-1, 0] = 1
                # mask[:-1, -1] = 1

                # Dam mask
                # mask[0, :] = 1
                # mask[-1, :] = 1
                # mask[:8, 8:10] = 1
                # mask[:, 0] = 1
                # mask[:, -1] = 1

                # Cavity mask
                # mask[0, :] = 1
                # mask[-1, :] = 1
                # mask[:-1, 0] = 1
                # mask[:-1, -1] = 1

                # import matplotlib.pyplot as plt
                # plt.imshow(mask)
                # plt.show()


                # print(cylinder_x, cylinder_y, cylinder_r)

                # import matplotlib.pyplot as plt

                # plt.imshow(mask)
                # plt.show()

                self.boundary.append(boundary)
                self.boundary_mask.append(boundary_mask)

            self.data_len += len(u_star)


        # Convert to torch tensors
        self.u_star = torch.tensor(np.array(self.u_star), dtype=torch.float32)
        self.phi = torch.tensor(np.array(self.phi), dtype=torch.float32)
        self.delta_x = torch.tensor(np.array(self.delta_x), dtype=torch.float32)
        self.delta_t = torch.tensor(np.array(self.delta_t), dtype=torch.float32)
        self.t = torch.tensor(np.array(self.t), dtype=torch.float32)
        self.rho = torch.tensor(np.array(self.rho), dtype=torch.float32)
        self.mu = torch.tensor(np.array(self.mu), dtype=torch.float32)
        self.boundary = torch.tensor(np.array(self.boundary), dtype=torch.float32)
        self.boundary_mask = torch.tensor(np.array(self.boundary_mask), dtype=torch.float32)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return NotImplementedError
    
    def get_dims(self):
        return self.input_channels, self.height, self.width, self.output_channels
    
class PoissonPINNDataset(PINNDataset):
    def __init__(self, data_path, N, samples_per_case=-1):
        super().__init__(data_path, N, samples_per_case)
        self.input_channels = 6
        self.output_channels = 1

    def __getitem__(self, idx):

        x = torch.cat([self.u_star[idx].permute(2, 0, 1), self.coords, self.boundary[idx].permute(2, 0, 1)], dim=0)

        return x, self.delta_x[idx], self.delta_y[idx], self.t[idx], self.delta_t[idx], self.rho[idx], self.mu[idx], self.coords, self.boundary_mask[idx]
    
    def get_phi(self, idx):
        return self.phi[idx]
    
class FullPINNDataset(PINNDataset):
    def __init__(self, data_path, N, samples_per_case=-1):
        super().__init__(data_path, N, samples_per_case)
        self.input_channels = 3
        self.output_channels = 3

    def __getitem__(self, idx):
        x = torch.cat([self.delta_t[idx], self.coords], dim=0)

        return x, self.delta_x[idx], self.delta_y[idx], self.t[idx], self.delta_t[idx], self.rho[idx], self.mu[idx], self.coords, self.boundary