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

    return u_star[:samples_per_case], phi[:samples_per_case], params["grid_spacing"], params["time_step"], params["density"], params["viscosity"]

class PINNDataset(Dataset):
    def __init__(self, data_path, N, samples_per_case=-1):
        # Initialize variables
        self.data_len = 0
        self.u_star = []
        self.phi = []
        self.delta_x = []
        self.delta_t = []
        self.t = []
        self.rho = []
        self.mu = []
        self.boundary = []

        # Iterate over cases
        for i in range(N):
            u_star, phi, grid_spacing, time_step, rho, mu = load_case(data_path, i, samples_per_case)

            # Save the case shape
            if i == 0:
                shape = np.shape(u_star)
                self.height, self.width = shape[1], shape[2]

                # Create a coordinate grid
                grid_x, grid_y = torch.meshgrid(
                        torch.linspace(0, self.width * grid_spacing, self.width),
                        torch.linspace(0, self.height * grid_spacing, self.height),
                        indexing="ij"
                )
                self.coords = torch.stack([grid_y, grid_x], dim=0)

            # Iterate over time steps
            for j in range(len(u_star)):
                self.u_star.append(u_star[j])
                self.phi.append(phi[j])
                self.delta_x.append(grid_spacing)
                self.rho.append(rho)
                self.mu.append(mu)

                t = torch.ones(self.height, self.width) * j * time_step
                t = t.unsqueeze(0)

                self.t.append(t)
                self.delta_t.append(time_step)

                # TODO: Figure out how to implement boundary
                self.boundary.append(np.nan)

            # Convert to torch tensors
            self.u_star = torch.tensor(np.array(self.u_star), dtype=torch.float32)
            self.phi = torch.tensor(np.array(self.phi), dtype=torch.float32)
            self.delta_x = torch.tensor(np.array(self.delta_x), dtype=torch.float32)
            self.delta_t = torch.tensor(np.array(self.delta_t), dtype=torch.float32)
            self.t = torch.tensor(np.array(self.t), dtype=torch.float32)
            self.rho = torch.tensor(np.array(self.rho), dtype=torch.float32)
            self.mu = torch.tensor(np.array(self.mu), dtype=torch.float32)
            self.boundary = torch.tensor(np.array(self.boundary), dtype=torch.float32)

            self.data_len += len(u_star)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return NotImplementedError
    
    def get_dims(self):
        return self.input_channels, self.height, self.width, self.output_channels
    
class PoissonPINNDataset(PINNDataset):
    def __init__(self, data_path, N, samples_per_case=-1):
        super().__init__(data_path, N, samples_per_case)
        self.input_channels = 4
        self.output_channels = 1

    def __getitem__(self, idx):
        x = torch.cat([self.u_star[idx].permute(2, 0, 1), self.coords], dim=0)

        return x, self.delta_x[idx], self.t[idx], self.delta_t[idx], self.rho[idx], self.mu[idx], self.coords, self.boundary
    
class FullPINNDataset(PINNDataset):
    def __init__(self, data_path, N, samples_per_case=-1):
        super().__init__(data_path, N, samples_per_case)
        self.input_channels = 3
        self.output_channels = 3

    def __getitem__(self, idx):
        x = torch.cat([self.delta_t[idx], self.coords], dim=0)

        return x, self.delta_x[idx], self.t[idx], self.delta_t[idx], self.rho[idx], self.mu[idx], self.coords, self.boundary