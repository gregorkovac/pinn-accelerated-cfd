import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import json

def load_case(data_path, idx):
    case_path = os.path.join(data_path, f"{idx}")

    u_star = np.load(os.path.join(case_path, "u_star.npy"))
    phi = np.load(os.path.join(case_path, "phi.npy"))

    with open(os.path.join(case_path, "params.json")) as f:
        params = json.load(f)

    return u_star, phi, params["grid_spacing"], params["time_step"], params["density"], params["viscosity"]

class PINNDataset(Dataset):
    def __init__(self, data_path, N):
        # Initialize variables
        self.data_len = 0
        self.u_star = []
        self.phi = []
        self.delta_x = []
        self.delta_t = []
        self.rho = []
        self.mu = []

        # Iterate over cases
        for i in range(N):
            u_star, phi, grid_spacing, time_step, rho, mu = load_case(data_path, i)

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
                self.delta_t.append(t)

            # Convert to torch tensors
            self.u_star = torch.tensor(np.array(self.u_star), dtype=torch.float32)
            self.phi = torch.tensor(np.array(self.phi), dtype=torch.float32)
            self.delta_x = torch.tensor(np.array(self.delta_x), dtype=torch.float32)
            self.delta_t = torch.tensor(np.array(self.delta_t), dtype=torch.float32)
            self.rho = torch.tensor(np.array(self.rho), dtype=torch.float32)
            self.mu = torch.tensor(np.array(self.mu), dtype=torch.float32)

            self.data_len += len(u_star)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return NotImplementedError
    
    def get_dims(self):
        return self.channels, self.height, self.width
    
class PoissonPINNDataset(PINNDataset):
    def __init__(self, data_path, N):
        super().__init__(data_path, N)
        self.channels = 4

    def __getitem__(self, idx):
        x = torch.cat([self.u_star[idx].permute(2, 0, 1), self.coords], dim=0).unsqueeze(0)

        return x, torch.tensor(self.delta_x[idx], dtype=torch.float32), torch.tensor(self.delta_t[idx], dtype=torch.float32), torch.tensor(self.rho[idx], dtype=torch.float32), torch.tensor(self.mu[idx], dtype=torch.float32), self.coords
    
class FullPINNDataset(PINNDataset):
    def __init__(self, data_path, N):
        super().__init__(data_path, N)
        self.channels = 3

    def __getitem__(self, idx):
        x = torch.cat([self.delta_t[idx], self.coords], dim=0).unsqueeze(0)

        return x, torch.tensor(self.delta_x[idx], dtype=torch.float32), torch.tensor(self.delta_t[idx], dtype=torch.float32), torch.tensor(self.rho[idx], dtype=torch.float32), torch.tensor(self.mu[idx], dtype=torch.float32), self.coords