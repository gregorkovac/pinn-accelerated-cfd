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

class PoissonPINNDataset(Dataset):
    def __init__(self, data_path, N):
        self.channels = 4

        self.data_len = 0

        xs = []
        ys = []
        grid_spacings = []
        rhos = []
        delta_ts = []

        grid_x, grid_y = torch.meshgrid(
                torch.linspace(0, self.width * grid_spacing, self.width),
                torch.linspace(0, self.height * grid_spacing, self.height),
                indexing="ij"
        )

        coords = torch.stack([grid_y, grid_x], dim=0)

        for i in range(N):

            u_star, phi, grid_spacing, delta_t, rho, _ = load_case(data_path, i)

            if i == 0:
                shape = np.shape(u_star)
                self.height, self.width = shape[1], shape[2]

            for j in range(len(u_star)):
                x = torch.stack([torch.tensor(u_star[j]).flatten().float(), coords], dim=0).unsqueeze(0)
                xs.append(x)
                # xs.append(torch.tensor(u_star[j]).flatten().float())
                ys.append(torch.tensor(phi[j]).flatten().float())
                grid_spacings.append(grid_spacing)
                delta_ts.append(delta_t)
                rhos.append(rho)

            self.data_len += len(u_star)


        self.x = torch.tensor(np.vstack(xs), dtype=torch.float32)
        self.y = torch.tensor(np.vstack(ys), dtype=torch.float32)
        self.delta_x = grid_spacings
        self.delta_t = delta_ts
        self.rho = rhos

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        u_star = self.x[idx].reshape(self.height, self.width, 2)
        u_star = u_star.permute(2, 0, 1)
        phi = self.y[idx].reshape(self.height, self.width)

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, self.width * self.delta_x[idx], self.width),
            torch.linspace(0, self.height * self.delta_x[idx], self.height),
            indexing="ij"
        )

        coords = torch.stack([grid_y, grid_x], dim=0)

        boundary = torch.zeros((1, self.height, self.width))
        boundary[:, 1:-1, 0] = 1

        return u_star, phi, torch.tensor(self.delta_x[idx], dtype=torch.float32), torch.tensor(self.delta_t[idx], dtype=torch.float32), torch.tensor(self.rho[idx], dtype=torch.float32), coords, boundary

    def get_dims(self):
        return self.channels, self.height, self.width

class FullPINNDataset(Dataset):
    def __init__(self, data_path, N):
        self.channels = 3

        self.data_len = 0

        xs = []
        ys = []
        grid_spacings = []
        rhos = []
        delta_ts = []
        mus = []

        for i in range(N):

            u_star, phi, grid_spacing, delta_t, rho, mu = load_case(data_path, i)

            if i == 0:
                shape = np.shape(u_star)
                self.height, self.width = shape[1], shape[2]

            grid_x, grid_y = torch.meshgrid(
                torch.linspace(0, self.width * grid_spacing, self.width),
                torch.linspace(0, self.height * grid_spacing, self.height),
                indexing="ij"
            )

            coords = torch.stack([grid_y, grid_x], dim=0)

            for j in range(len(u_star)):

                time_step = torch.ones(u_star.shape[1], u_star.shape[2]) * j * delta_t
                time_step = time_step.unsqueeze(0)

                xs.append(torch.cat([time_step, coords], dim=0).unsqueeze(0))

                grid_spacings.append(grid_spacing)
                delta_ts.append(delta_t)
                rhos.append(rho)
                mus.append(mu)

            self.data_len += len(u_star)


        self.x = torch.tensor(np.vstack(xs), dtype=torch.float32)
        self.delta_x = grid_spacings
        self.delta_t = delta_ts
        self.rho = rhos
        self.mu = mus

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        x = self.x[idx].reshape(3, self.height, self.width)

        return x, torch.empty(0), torch.tensor(self.delta_x[idx], dtype=torch.float32), torch.tensor(self.delta_t[idx], dtype=torch.float32), torch.tensor(self.rho[idx], dtype=torch.float32), torch.tensor(self.mu[idx], dtype=torch.float32), torch.empty(0)

    def get_dims(self):
        return self.channels, self.height, self.width