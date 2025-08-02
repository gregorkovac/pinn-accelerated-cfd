import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import json

def load_case(idx):
    DATA_PATH = "./numeric_method/generated_data_base_2"
    case_path = os.path.join(DATA_PATH, f"{idx}")

    u_star = np.load(os.path.join(case_path, "u_star.npy"))
    # div_u_star = np.load(os.path.join(case_path, "div_u_star.npy"))
    phi = np.load(os.path.join(case_path, "phi.npy"))

    with open(os.path.join(case_path, "params.json")) as f:
        params = json.load(f)

    return u_star[:30], phi[:30], params["grid_spacing"], params["time_step"], params["density"]

class PoissonPINNDataset(Dataset):
    def __init__(self, N=100, width=20, height=20):
        self.data_len = 0

        xs = []
        # div_u_stars = []
        ys = []
        grid_spacings = []
        rhos = []
        delta_ts = []

        self.width = width
        self.height = height

        for i in range(N):

            u_star,  phi, grid_spacing, delta_t, rho = load_case(i)

            for j in range(len(u_star)):
                xs.append(torch.tensor(u_star[j]).flatten().float())
                # div_u_stars.append(torch.tensor(div_u_star[j]).flatten().float())
                ys.append(torch.tensor(phi[j]).flatten().float())
                grid_spacings.append(grid_spacing)
                delta_ts.append(delta_t)
                rhos.append(rho)

            self.data_len += len(u_star)


        self.x = torch.tensor(np.vstack(xs), dtype=torch.float32)
        self.y = torch.tensor(np.vstack(ys), dtype=torch.float32)
        # self.div_u_star = torch.tensor(np.vstack(div_u_stars), dtype=torch.float32)
        self.delta_x = grid_spacings
        self.delta_t = delta_ts
        self.rho = rhos

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        u_star = self.x[idx].reshape(self.height, self.width, 2)
        u_star = u_star.permute(2, 0, 1)
        # div_u_star = self.div_u_star[idx].reshape(1, self.height, self.width)
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

class FullPINNDataset(Dataset):
    def __init__(self, N=100, width=20, height=20):
        self.data_len = 0

        xs = []
        ys = []
        grid_spacings = []
        rhos = []
        delta_ts = []
        mus = []

        self.width = width
        self.height = height

        for i in range(N):

            u_star, phi, grid_spacing, delta_t, rho, mu = load_case(i)

            grid_x, grid_y = torch.meshgrid(
                torch.linspace(0, self.width * grid_spacing, self.width),
                torch.linspace(0, self.height * grid_spacing, self.height),
                indexing="ij"
            )

            coords = torch.stack([grid_y, grid_x], dim=0)
            bc = torch.zeros(3, u_star.shape[1], u_star.shape[2])

            # Y velocity BC
            bc[0, 0, :] = -1
            bc[0, -1, :] = -1

            # X velocity BC
            bc[1, :, 0] = 1
            bc[1, 0, :] = -1
            bc[1, -1, :] = -1

            # Pressure BC
            bc[2, :, -1] = -1
            bc[2, 0, :] = 0
            bc[2, -1, :] = 0

            for j in range(len(u_star)):
                # xs.append(torch.tensor(u_star[j]).flatten().float())

                time_step = torch.ones(u_star.shape[1], u_star.shape[2]) * j * delta_t
                time_step = time_step.unsqueeze(0)

                xs.append(torch.cat([time_step, coords], dim=0).unsqueeze(0))

                # div_u_stars.append(torch.tensor(div_u_star[j]).flatten().float())
                # ys.append(torch.tensor(phi[j]).flatten().float())
                grid_spacings.append(grid_spacing)
                delta_ts.append(delta_t)
                rhos.append(rho)
                mus.append(mu)

            self.data_len += len(u_star)


        self.x = torch.tensor(np.vstack(xs), dtype=torch.float32)
        # self.y = torch.tensor(np.vstack(ys), dtype=torch.float32)
        # self.div_u_star = torch.tensor(np.vstack(div_u_stars), dtype=torch.float32)
        self.delta_x = grid_spacings
        self.delta_t = delta_ts
        self.rho = rhos
        self.mu = mus

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        x = self.x[idx].reshape(3, self.height, self.width)
        # print(x.shape)
        # fig, ax = plt.subplots(figsize=(8, 8))
        # plt.ioff()
        # ax.imshow(x[1])
        # plt.show()

        # exit(0)
        # x = x.permute(2, 0, 1)
        # div_u_star = self.div_u_star[idx].reshape(1, self.height, self.width)
        # phi = self.y[idx].reshape(self.height, self.width)

        # # grid_x, grid_y = torch.meshgrid(
        # #     torch.linspace(0, self.width * self.delta_x[idx], self.width),
        # #     torch.linspace(0, self.height * self.delta_x[idx], self.height),
        # #     indexing="ij"
        # # )

        # # coords = torch.stack([grid_y, grid_x], dim=0)

        # boundary = torch.zeros((1, self.height, self.width))
        # boundary[:, 1:-1, 0] = 1

        return x, torch.tensor(self.delta_x[idx], dtype=torch.float32), torch.tensor(self.delta_t[idx], dtype=torch.float32), torch.tensor(self.rho[idx], dtype=torch.float32), torch.tensor(self.mu[idx], dtype=torch.float32)
