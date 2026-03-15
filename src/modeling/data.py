import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
import os
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

def load_case(data_path, idx, samples_per_case=-1):
    case_path = os.path.join(data_path, f"{idx}")

    u_star = np.load(os.path.join(case_path, "u_star.npy"))
    div_u_star = np.load(os.path.join(case_path, "div_u_star.npy"))
    phi = np.load(os.path.join(case_path, "phi.npy"))

    u_star = u_star #[:100]
    phi = phi #[:100]
    div_u_star = div_u_star #[:100]

    with open(os.path.join(case_path, "params.json")) as f:
        params = json.load(f)

    if samples_per_case == -1:
        samples_per_case = len(u_star)

    indices = np.random.choice(samples_per_case, samples_per_case, replace=False)

    return u_star[indices], phi[indices], params["dx"], params["dy"], params["time_step"], params["density"], params["viscosity"], params["geometry"], div_u_star[indices] #, params["cylinder_x"], params["cylinder_y"], params["cylinder_r"]

def get_geometry_region(g, width, height):
    # Extract the region
    x, y = g["pos"]["x"], g["pos"]["y"]
    w, h = g["size"]["width"], g["size"]["height"]

    # The normal is defined either by a direction or the velocity of the inflow
    if "dir" in g.keys():
        xn, yn = g["dir"]["x"], g["dir"]["y"]
    elif "vel" in g.keys():
        xn, yn = g["vel"]["x"], g["vel"]["y"]
    else:
        xn, yn = None, None

    # Convert the "-1" notation to width/height
    if x == -1:
        x = width - 1
    if y == -1:
        y = height - 1
    if w == -1:
        w = width - x
    if h == -1:
        h = height - y

    return x, y, w, h, xn, yn

def geometry_to_mtx(geometry, width, height):
    # Initialize tensor for storing normals for boundary conditions on pressure and velocity
    p_dirichlet = np.zeros((height, width))
    p_dirichlet_mask = np.zeros((height, width))
    p_neumann = np.zeros((height, width, 2))

    u_dirichlet = np.zeros((height, width, 2))
    u_dirichlet_mask = np.zeros((height, width, 2))
    u_neumann = np.zeros((height, width, 2))

    # Initialize a binary mask for storing at which points the boundary is present
    boundary_mask = np.zeros((height, width))

    cylinder_mask = None

    for g in geometry:
        if g["type"] != "cylinder":
            x, y, w, h, xn, yn = get_geometry_region(g, width, height)

        if g["type"] == "inflow":
            boundary_mask[y:(y+h), x:(x+w)] = 1

            u_dirichlet[y:(y+h), x:(x+w)] = np.array([xn, yn])
            u_dirichlet_mask[y:(y+h), x:(x+w)] = 1

            # p_neumann[y:(y+h), x:(x+w)] = np.array([xn != 0, yn != 0], dtype=int)
            p_neumann[y:(y+h), x:(x+w)] = np.array([(xn != 0) * np.sign(xn), (yn != 0) + np.sign(yn)], dtype=int)
        elif g["type"] == "wall":
            boundary_mask[y:(y+h), x:(x+w)] = 1

            u_dirichlet[y:(y+h), x:(x+w)] = np.array([0, 0])
            u_dirichlet_mask[y:(y+h), x:(x+w)] = 1

            # p_neumann[y:(y+h), x:(x+w)] = np.array([xn != 0, yn != 0], dtype=int)
            p_neumann[y:(y+h), x:(x+w)] = np.array([xn, yn], dtype=int)
        elif g["type"] == "outflow":
            boundary_mask[y:(y+h), x:(x+w)] = 1

            p_dirichlet[y:(y+h), x:(x+w)] = 0
            p_dirichlet_mask[y:(y+h), x:(x+w)] = 1

            u_neumann[y:(y+h), x:(x+w)] = np.array([xn, yn])
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
            u_dirichlet_mask[y_min:y_max, x_min:x_max, :][mask] = 1

    return u_dirichlet, u_dirichlet_mask, u_neumann, p_dirichlet, p_dirichlet_mask, p_neumann, boundary_mask

class PINNDataset(Dataset):
    def __init__(self, data_path, N, device, samples_per_case=-1, start_sample=0):
        if data_path == None:
            self.height = 0
            self.width = 0
            return

        self.device = device

        # Initialize variables
        self.data_len = 0
        self.u_star = []
        self.phi = []
        self.delta_x = []
        self.delta_y = []
        self.delta_t = []
        self.rho = []
        self.mu = []
        self.u_dirichlet = []
        self.u_dirichlet_mask = []
        self.u_neumann = []
        self.p_dirichlet = []
        self.p_dirichlet_mask = []
        self.p_neumann = []
        self.boundary_mask = []
        self.div_u_star = []

        print("Loading data...")

        # Iterate over cases
        for i in tqdm(range(start_sample, start_sample + N)):
            u_star, phi, dx, dy, time_step, rho, mu, geometry, div_u_star = load_case(data_path, i, samples_per_case)

            # Save the case shape
            if i == start_sample:
                shape = np.shape(u_star)
                self.height, self.width = shape[1], shape[2]

                # Create a coordinate grid
                grid_y, grid_x = torch.meshgrid(
                        torch.linspace(0, 1, self.height),
                        torch.linspace(0, 1, self.width),
                        indexing="xy"
                )

                # plt.ioff()
                # fig, ax = plt.subplots(1, 2)
                # im0 = ax[0].imshow(grid_y)
                # im1 = ax[1].imshow(grid_x)

                # plt.colorbar(im0)
                # plt.colorbar(im1)

                # plt.show()

                self.coords = torch.stack([grid_x, grid_y], dim=0).to(device)

            u_dirichlet, u_dirichlet_mask, u_neumann, p_dirichlet, p_dirichlet_mask, p_neumann, boundary_mask = geometry_to_mtx(geometry, self.width, self.height)
        
            # Iterate over time steps
            for j in range(len(u_star)):
                
                # Fields of interest 
                self.u_star.append(u_star[j])
                self.div_u_star.append(div_u_star[j])
                self.phi.append(phi[j])
                
                # Time-space parameters
                self.delta_x.append(dx)
                self.delta_y.append(dy)
                self.delta_t.append(time_step)

                # Fluid parameters
                self.rho.append(rho)
                self.mu.append(mu)

                # Boundary conditions
                self.u_dirichlet.append(u_dirichlet)
                self.u_dirichlet_mask.append(u_dirichlet_mask)
                self.u_neumann.append(u_neumann)
                self.p_dirichlet.append(p_dirichlet)
                self.p_dirichlet_mask.append(p_dirichlet_mask)
                self.p_neumann.append(p_neumann)
                self.boundary_mask.append(boundary_mask)

            self.data_len += len(u_star)


        # Convert to torch tensors
        self.u_star = torch.tensor(np.array(self.u_star, dtype=np.float32), dtype=torch.float32, device=device)
        self.div_u_star = torch.tensor(np.array(self.div_u_star, dtype=np.float32), dtype=torch.float32, device=device)
        self.phi = torch.tensor(np.array(self.phi, dtype=np.float32), dtype=torch.float32, device=device)

        self.delta_x = torch.tensor(np.array(self.delta_x, dtype=np.float32), dtype=torch.float32, device=device)
        self.delta_y = torch.tensor(np.array(self.delta_y, dtype=np.float32), dtype=torch.float32, device=device)
        self.delta_t = torch.tensor(np.array(self.delta_t, dtype=np.float32), dtype=torch.float32, device=device)

        self.rho = torch.tensor(np.array(self.rho, dtype=np.float32), dtype=torch.float32, device=device)
        self.mu = torch.tensor(np.array(self.mu, dtype=np.float32), dtype=torch.float32, device=device)

        self.u_dirichlet = torch.tensor(np.array(self.u_dirichlet, dtype=np.float32), dtype=torch.float32, device=device)
        self.u_dirichlet_mask = torch.tensor(np.array(self.u_dirichlet_mask, dtype=np.float32), dtype=torch.float32, device=device)
        self.u_neumann = torch.tensor(np.array(self.u_neumann, dtype=np.float32), dtype=torch.float32, device=device)
        self.p_dirichlet = torch.tensor(np.array(self.p_dirichlet, dtype=np.float32), dtype=torch.float32, device=device)
        self.p_dirichlet_mask = torch.tensor(np.array(self.p_dirichlet_mask, dtype=np.float32), dtype=torch.float32, device=device)
        self.p_neumann = torch.tensor(np.array(self.p_neumann, dtype=np.float32), dtype=torch.float32, device=device)
        self.boundary_mask = torch.tensor(np.array(self.boundary_mask, dtype=np.float32), dtype=torch.float32, device=device)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return NotImplementedError
    
    def get_dims(self):
        return self.input_channels, self.height, self.width, self.output_channels
    
    def enforceBC(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

class PoissonPINNDataset(PINNDataset):
    def __init__(self, data_path, N, device, samples_per_case=-1, start_sample=0):
        super().__init__(data_path, N, device, samples_per_case, start_sample)
        self.input_channels = 3
        self.output_channels = 1

        if N == 0:
            return

        self.x = []
        for idx in range(N * samples_per_case):


            shp = self.p_dirichlet[idx].unsqueeze(0).shape
            ones = torch.ones(shp).to(device)

            # self.x.append(torch.cat([self.u_star[idx].permute(2, 0, 1), # 0, 1
            #            self.coords, # 2, 3
            #            self.p_neumann[idx].permute(2, 0, 1), # 4, 5
            #            self.p_dirichlet[idx].unsqueeze(0), # 6
            #            self.p_dirichlet_mask[idx].unsqueeze(0), # 7
            #            self.rho[idx] * ones, # 8
            #            self.delta_t[idx] * ones, # 9 
            #            self.boundary_mask[idx].unsqueeze(0), # 10
            #            self.div_u_star[idx].unsqueeze(0) # 11
            #         ], dim=0))

            # self.x.append(torch.cat([
            #                self.div_u_star[idx].unsqueeze(0), #*(1 - self.boundary_mask[idx].unsqueeze(0)), # 0
            #                0 * self.coords, # 1, 2
            #                0 * self.p_dirichlet[idx].unsqueeze(0), # 3,
            #                0 * self.p_dirichlet_mask[idx].unsqueeze(0), # 4
            #                self.boundary_mask[idx].unsqueeze(0), # 5
            #                0 * self.u_star[idx].permute(2, 0, 1), # 6, 7
            #                0 * self.p_neumann[idx].permute(2, 0, 1), # 8, 9
            #                self.rho[idx] * ones, # 10
            #                self.delta_t[idx] * ones, # 11
            #                self.delta_x[idx] * ones, # 12
            #                self.delta_y[idx] * ones, # 13
            #             ], dim=0))

            # self.x.append(torch.cat([
            #                self.div_u_star[idx].unsqueeze(0), #*(1 - self.boundary_mask[idx].unsqueeze(0)), # 0
            #                self.boundary_mask[idx].unsqueeze(0), # 1
            #                self.rho[idx] * ones, # 2
            #                self.delta_t[idx] * ones, # 3
            #                self.delta_x[idx] * ones, # 4
            #                self.delta_y[idx] * ones, # 5
            #             ], dim=0))

            self.x.append(torch.cat([
                           self.div_u_star[idx].unsqueeze(0), #*(1 - self.boundary_mask[idx].unsqueeze(0)), # 0
                           self.delta_x[idx] * ones, # 1
                           self.delta_y[idx] * ones, # 2
                        ], dim=0))

            # self.x.append(torch.cat([
            #                self.div_u_star[idx].unsqueeze(0), #*(1 - self.boundary_mask[idx].unsqueeze(0)), # 0
            #                self.rho[idx] * ones, # 1
            #                self.delta_t[idx] * ones, # 2
            #                self.delta_x[idx] * ones, # 3
            #                self.delta_y[idx] * ones, # 4
            #             ], dim=0))
            
            # self.x.append(torch.cat([
            #                self.div_u_star[idx].unsqueeze(0), #*(1 - self.boundary_mask[idx].unsqueeze(0)), # 0
            #                self.delta_x[idx] * ones, # 1
            #                self.delta_y[idx] * ones, # 2
            #             #    self.coords # 3, 4
            #             ], dim=0))

            # plt.ioff()
            # fig, ax = plt.subplots(3, 4)
            # ax = ax.flatten()
            # for j in range(0, 12):
            #     im = ax[j].imshow(self.x[-1][j].cpu().detach().numpy())
            #     plt.colorbar(im)

            # plt.show()

        self.x = torch.stack(self.x, dim=0).to(device)

        # for k in range(self.x.shape[1]):
        #     print(f"{k}  :    {self.x[0, k].mean()} +- {self.x[0, k].std()}      min = {self.x[0, k].min()}   max = {self.x[0, k].max()}")
        # exit(0)


    def __getitem__(self, idx):
        return self.x[idx], self.delta_x[idx], self.delta_y[idx], self.delta_t[idx], self.rho[idx], self.mu[idx], self.coords, self.boundary_mask[idx]
    
    def get_phi(self, idx):
        return self.phi[idx]
    
    def enforceBC(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:

        # out[..., 1:-1, -1] = 0

        # out[..., 0, :] = 0
        # out[..., -1, :] = 0
        # out[..., 0] = 0
        # out[..., -1] = 0

        return out

        p_dirichlet = x[:, 3:4]
        p_dirichlet_mask = x[:, 4:5]
        p_neumann = x[:, 8:10]
        boundary_mask = x[:, 5:6]

        p = out
        # p = F.pad(out[..., 1:-1, 1:-1], (1, 1, 1, 1), mode="replicate")
        # p = F.pad(out, (1, 1, 1, 1), mode="constant", value=0)
        # p = F.pad(out, (1, 1, 1, 1), mode="replicate")
        # p = F.pad(out, (1, 1, 1, 1), mode="reflect")

        # p_dirichlet_mask = F.pad(p_dirichlet_mask[:, :, 1:-1, 1:-1], (1, 1, 1, 1), mode="constant", value=1)
        # p_dirichlet_mask = F.pad(p_dirichlet_mask, (1, 1, 1, 1), mode="constant", value=1)
        # p_dirichlet = F.pad(p_dirichlet, (1, 1, 1, 1), mode="constant", value=0)
        # p_neumann = F.pad(p_neumann, (1, 1, 1, 1), mode="constant", value=1)
        # boundary_mask = F.pad(boundary_mask, (1, 1, 1, 1), mode="constant", value=1)

        # plt.ioff()
        # plt.imshow(p_dirichlet_mask[0, 0].detach().cpu().numpy())
        # plt.show()

        # exit(0)

        # print(f"Inside enforce BC after pad: {p.shape}")

        # p = torch.where((boundary_mask == 1) & (p_neumann[:, 1].unsqueeze(1) == -1), torch.roll(p, 1, dims=-2), p)
        # p = torch.where((boundary_mask == 1) & (p_neumann[:, 1].unsqueeze(1) == 1), torch.roll(p, -1, dims=-2), p)

        # p = torch.where((boundary_mask == 1) & (p_neumann[:, 0].unsqueeze(1) == -1), torch.roll(p, 1, dims=-1), p)
        # p = torch.where((boundary_mask == 1) & (p_neumann[:, 0].unsqueeze(1) == 1), torch.roll(p, -1, dims=-1), p)

        # p = torch.where((boundary_mask == 1) & (p_neumann[:, 1].unsqueeze(1) == -1), torch.roll(p, 1, dims=-2), p)
        # p = torch.where((boundary_mask == 1) & (p_neumann[:, 1].unsqueeze(1) == 1), torch.roll(p, -1, dims=-2), p)


        p = p * (1 - p_dirichlet_mask) + p_dirichlet * p_dirichlet_mask

        # plt.ioff()
        # plt.imshow(p[0, 0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        
        return p

class FullPINNDataset(PINNDataset):
    def __init__(self, data_path, N, device, samples_per_case=-1, start_sample=0):
        super().__init__(data_path, N, device, samples_per_case, start_sample)
        self.input_channels = 17
        self.output_channels = 3

        if N == 0:
            return

        self.x = []

        for idx in range(N * samples_per_case):
            shp = self.p_dirichlet[idx].unsqueeze(0).shape
            ones = torch.ones(shp).to(device)

            self.x.append(torch.cat([
                    #    F.pad(self.coords, (1, 1, 1, 1)), # 0, 1
                       0 * self.coords,
                       self.u_dirichlet[idx].permute(2, 0, 1), # 2, 3
                       self.u_dirichlet_mask[idx].permute(2, 0, 1), # 4, 5
                       self.u_neumann[idx].permute(2, 0, 1), # 6, 7
                       self.p_dirichlet[idx].unsqueeze(0), # 8
                       self.p_dirichlet_mask[idx].unsqueeze(0), # 9
                       self.p_neumann[idx].permute(2, 0, 1), # 10, 11
                       self.rho[idx] * ones, # 12
                       self.mu[idx] * ones, # 13
                       self.boundary_mask[idx].unsqueeze(0), # 14
                       self.delta_x[idx] * ones, # 15
                       self.delta_y[idx] * ones # 16
                    ], dim=0) )

        self.x = torch.stack(self.x, dim=0).to(device)

    def __getitem__(self, idx):

        return self.x[idx], self.delta_x[idx], self.delta_y[idx], self.delta_t[idx], self.rho[idx], self.mu[idx], self.coords, self.boundary_mask[idx]

        # x = torch.cat([self.coords, # 0, 1
        #                self.u_dirichlet[idx].permute(2, 0, 1), # 2, 3
        #                self.u_dirichlet_mask[idx].permute(2, 0, 1), # 4, 5
        #                self.u_neumann[idx].permute(2, 0, 1), # 6, 7
        #                self.p_dirichlet[idx].unsqueeze(0), # 8
        #                self.p_dirichlet_mask[idx].unsqueeze(0), # 9
        #                self.p_neumann[idx].permute(2, 0, 1), # 10, 11
        #             ], dim=0) 

        # return x, self.delta_x[idx], self.delta_y[idx], self.t[idx], self.delta_t[idx], self.rho[idx], self.mu[idx], self.coords, self.boundary_mask[idx]
    
    def enforceBC(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        u_dirichlet = x[:, 2:4]
        u_dirichlet_mask = x[:, 4:6] #.expand(-1, 2, -1, -1)
        u_neumann = x[:, 6:8]
        p_dirichlet = x[:, 8:9]
        p_dirichlet_mask = x[:, 9:10]
        p_neumann = x[:, 10:12]
        boundary_mask = x[:, 14:15]


        # u_dirichlet_mask = F.pad(u_dirichlet_mask, (1, 1, 1, 1), mode="constant", value=1)
        # u_dirichlet = F.pad(u_dirichlet, (1, 1, 1, 1), mode="constant", value=0)
        # u_neumann = F.pad(u_neumann, (1, 1, 1, 1), mode="constant", value=1)
        # p_dirichlet_mask = F.pad(p_dirichlet_mask, (1, 1, 1, 1), mode="constant", value=1)
        # p_dirichlet = F.pad(p_dirichlet, (1, 1, 1, 1), mode="constant", value=0)
        # p_neumann = F.pad(p_neumann, (1, 1, 1, 1), mode="constant", value=1)
        # boundary_mask = F.pad(boundary_mask, (1, 1, 1, 1), mode="constant", value=1)

        # plt.ioff()
        # fig, ax = plt.subplots(2, 6)
        # ax = ax.flatten()

        # im0 = ax[0].imshow(u_dirichlet[0, 0].detach().cpu().numpy())
        # ax[0].set_title("u_dirichlet[0]")
        # plt.colorbar(im0)

        # im1 = ax[1].imshow(u_dirichlet[0, 1].detach().cpu().numpy())
        # ax[1].set_title("u_dirichlet[1]")
        # plt.colorbar(im1)

        # im2 = ax[2].imshow(u_dirichlet_mask[0, 0].detach().cpu().numpy())
        # ax[2].set_title("u_dirichlet_mask[0]")
        # plt.colorbar(im2)

        # im3 = ax[3].imshow(u_dirichlet_mask[0, 1].detach().cpu().numpy())
        # ax[3].set_title("u_dirichlet_mask[0]")
        # plt.colorbar(im3)

        # im4 = ax[4].imshow(u_neumann[0, 0].detach().cpu().numpy())
        # ax[4].set_title("u_neumann[0]")
        # plt.colorbar(im4)

        # im5 = ax[5].imshow(u_neumann[0, 1].detach().cpu().numpy())
        # ax[5].set_title("u_neumann[1]")
        # plt.colorbar(im5)

        # im6 = ax[6].imshow(p_dirichlet[0, 0].detach().cpu().numpy())
        # ax[6].set_title("p_dirichlet")
        # plt.colorbar(im6)

        # im7 = ax[7].imshow(p_dirichlet_mask[0, 0].detach().cpu().numpy())
        # ax[7].set_title("p_dirichlet_mask")
        # plt.colorbar(im7)

        # im8 = ax[8].imshow(p_neumann[0, 0].detach().cpu().numpy())
        # ax[8].set_title("p_neumann[0]")
        # plt.colorbar(im8)

        # im9 = ax[9].imshow(p_neumann[0, 1].detach().cpu().numpy())
        # ax[9].set_title("p_neumann[1]")
        # plt.colorbar(im9)

        # im10 = ax[10].imshow(out[0, 0].detach().cpu().numpy())
        # ax[10].set_title("u[0]")
        # plt.colorbar(im10)

        # plt.show()
        # out = F.pad(out, (1, 1, 1, 1), mode="replicate")

        u = out[:, :2]
        p = out[:, 2:3]



        # vmin = -0.01
        # vmax = 0.01
        
        # plt.ioff()
        # fig, ax = plt.subplots(3, 2)
        # ax = ax.flatten()
        # im0 = ax[0].imshow(u[0, 0].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
        # im1 = ax[1].imshow(u[0, 1].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
        # ax[0].set_title("Original")
        # ax[1].set_title("Original")
        # plt.colorbar(im0)
        # plt.colorbar(im1)

        # u[..., -1] = u[..., -2]
        # u[:, -1, :] = 10

        u = torch.where((boundary_mask == 1) & (u_neumann[:, 0].unsqueeze(1) == -1), torch.roll(u, 1, dims=-1), u)
        u = torch.where((boundary_mask == 1) & (u_neumann[:, 0].unsqueeze(1) == 1), torch.roll(u, -1, dims=-1), u)

        u = torch.where((boundary_mask == 1) & (u_neumann[:, 1].unsqueeze(1) == -1), torch.roll(u, 1, dims=-2), u)
        u = torch.where((boundary_mask == 1) & (u_neumann[:, 1].unsqueeze(1) == 1), torch.roll(u, -1, dims=-2), u)

        p = torch.where((boundary_mask == 1) & (p_neumann[:, 0].unsqueeze(1) == -1), torch.roll(p, 1, dims=-1), p)
        p = torch.where((boundary_mask == 1) & (p_neumann[:, 0].unsqueeze(1) == 1), torch.roll(p, -1, dims=-1), p)

        p = torch.where((boundary_mask == 1) & (p_neumann[:, 1].unsqueeze(1) == -1), torch.roll(p, 1, dims=-2), p)
        p = torch.where((boundary_mask == 1) & (p_neumann[:, 1].unsqueeze(1) == 1), torch.roll(p, -1, dims=-2), p)

        # u = torch.where(
        #     u_neumann[:, 1].unsqueeze(1) == -1, torch.roll(u, 1, dims=2),
        #     torch.where(u_neumann[:, 1].unsqueeze(1) ==  1, torch.roll(u, -1, dims=2), u)
        # )

        # im2 = ax[2].imshow(u[0, 0].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
        # im3 = ax[3].imshow(u[0, 1].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
        # ax[2].set_title("After Neumann")
        # ax[3].set_title("After Neumann")
        # plt.colorbar(im2)
        # plt.colorbar(im3)


        # u_y = u[:, 0] * (1 - u_dirichlet_mask) + u_dirichlet[:, 0] * u_dirichlet_mask
        # u_x = u[:, 1] * (1 - u_dirichlet_mask) + u_dirichlet[:, 1] * u_dirichlet_mask
        u = u * (1 - u_dirichlet_mask) + u_dirichlet * u_dirichlet_mask
        p = p * (1 - p_dirichlet_mask) + p_dirichlet * p_dirichlet_mask

        # im4 = ax[4].imshow(u[0, 0].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
        # im5 = ax[5].imshow(u[0, 1].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
        # ax[4].set_title("After Dirichlet")
        # ax[5].set_title("After Dirichlet")
        # plt.colorbar(im4)
        # plt.colorbar(im5)
        # plt.tight_layout()
        # plt.show()


        # plt.ioff()
        # fig, ax = plt.subplots(1, 3)
        # im0 = ax[0].imshow(u[0, 0].detach().cpu().numpy())
        # im1 = ax[1].imshow(u[0, 1].detach().cpu().numpy())
        # im2 = ax[2].imshow(p[0, 0].detach().cpu().numpy())
        # plt.colorbar(im0)
        # plt.colorbar(im1)
        # plt.colorbar(im2)

        # ax[0].set_title("ux")
        # ax[1].set_title("uy")
        # ax[2].set_title("p")
        # plt.show()

        return torch.cat([u, p], dim=1).to(u.device)

class CaseBatchSampler(Sampler):
    def __init__(self, dataset_len, samples_per_case):
        self.dataset_len = dataset_len
        self.samples_per_case = samples_per_case
        self.num_cases = dataset_len // samples_per_case

    def __iter__(self):
        case_order = list(range(self.num_cases))
        random.shuffle(case_order)
        for case_idx in case_order:
            start = case_idx * self.samples_per_case
            end = start + self.samples_per_case
            yield list(range(start, end))

    def __len__(self):
        return self.num_cases