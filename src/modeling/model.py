import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels=2, output_channels=1, num_layers=3, hidden_channels=64, input_width=20, input_height=20, enforceBC=True):
        super(CNN, self).__init__()
        
        self.input_width = input_width
        self.input_height = input_height

        self.output_channels = output_channels

        self.enforceBC = enforceBC

        encoder_layers = []

        for i in range(num_layers):
            in_channels = input_channels if i == 0 else hidden_channels

            encoder_layers.extend([
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="zeros"),
                # nn.BatchNorm2d(hidden_channels),
                nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
                # nn.LayerNorm([hidden_channels, input_height, input_width]),
                nn.Tanh()
            ])

        self.encoder = nn.Sequential(*encoder_layers)

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1, padding_mode="reflect")
        # )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, output_channels, 3, padding=1, padding_mode="zeros")
        )

        self.skip = nn.Conv2d(4, output_channels, 1)

        # self.bc_skip = nn.Conv2d(3, output_channels, 1)

        # self.bc_corrector = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, padding=1),
        #     nn.Tanh(),
        #     nn.Conv2d(16, 2, 3, padding=1)
        # )

        self.initialize_weights()

    def forward(self, x):

        # mask = x[:, 4]

        u_dirichlet = x[:, 3:5]
        u_dirichlet_mask = x[:, 5]

        sdf = x[:, -1].unsqueeze(1)

        x_skip = torch.cat([sdf, u_dirichlet, u_dirichlet_mask.unsqueeze(1)], dim=1)

        skip = self.skip(x_skip)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(skip[0, 2].detach())
        # plt.show()
        # plt.ion()

        x = self.encoder(x)
        x = self.decoder(x)

        # bc_input = torch.cat([u_dirichlet, u_dirichlet_mask.unsqueeze(1)], dim=1)
        # bc_correction = self.bc_corrector(bc_input)

        # boundary_weight = u_dirichlet_mask.unsqueeze(1)
        # smooth_mask = torch.sigmoid(100 * (boundary_weight - 0.5))

        # print(smooth_mask.shape)

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow(boundary_weight[0, 0].detach())
        # plt.show()

        # x[:, :2] *= (1 - boundary_weight) 
        # x[:, :2] += bc_correction * boundary_weight

        # print(u_dirichlet.shape, u_dirichlet_mask.shape)

        # bc_feat = torch.cat([u_dirichlet, u_dirichlet_mask.unsqueeze(1)], dim=1)
        # skip = self.bc_skip(bc_feat) 

        x += skip

        # x = x * 0

        # import matplotlib.pyplot as plt
        # plt.ioff()
        # plt.imshow((1 - u_dirichlet_mask).abs()[4].detach())
        # plt.show()

        # x[:, :2][u_dirichlet_mask.unsqueeze(1).expand(-1, 2, -1, -1) == 1] = u_dirichlet[u_dirichlet_mask.unsqueeze(1).expand(-1, 2, -1, -1) == 1]
 
        # x = x + skip

        # if self.output_channels == 1:
        x = x.squeeze(1)

        if self.enforceBC:

            u_dirichlet_mask = u_dirichlet_mask.unsqueeze(1).expand(-1, 2, self.input_height, self.input_width)

            nonzero_mask = (u_dirichlet != 0).float()
            nonzero_mask = nonzero_mask.any(dim=1).unsqueeze(1)
            # nonzero_mask = 1 - nonzero_mask
            # print(nonzero_mask.shape)
            # nonzero_mask = nonzero_mask.unsqueeze(1).expand(-1, 2, self.input_height, self.input_width)
    

            # # print(u_dirichlet_mask.shape)
            # # print(nonzero_mask.shape)
            # u_dirichlet_mask = u_dirichlet_mask * nonzero_mask
            # print(u_dirichlet_mask.shape)

            x[:, :2] *= (1 - u_dirichlet_mask)
            x[:, :2] += u_dirichlet_mask * u_dirichlet


            # k = 10
            # alpha = torch.exp(-k * torch.abs(sdf))

            # import matplotlib.pyplot as plt
            # plt.ioff()
            # plt.imshow(u_dirichlet_mask[0, 1].detach())
            # plt.show()

            # x[:, :2] *= (1 - alpha)
            # x[:, :2] += alpha * u_dirichlet

            
        # x[:, :2] = (1 - u_dirichlet_mask).abs() * x[:, :2] + u_dirichlet_mask * u_dirichlet

        # # x[:, :2] = (1 - u_dirichlet_mask).abs() * x[:, :2] + u_dirichlet_mask * u_dirichlet
        # # print((1 - u_dirichlet_mask).abs().shape)
        # # print(x[:, :2].shape)

        # print(x.shape)
        # print(u_dirichlet_mask.shape)
        # print(u_dirichlet.shape)

        # exit(0)

        # x[u_dirichlet_mask == 1] = u_dirichlet

        # x = x * abs(1 - mask)

        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # with torch.no_grad():
        #     self.bc_skip.weight.zero_()
        #     self.bc_skip.weight[0, 0, 0, 0] = 1.0
        #     self.bc_skip.weight[1, 1, 0, 0] = 1.0
        #     if self.bc_skip.bias is not None:
        #         self.bc_skip.bias.zero_()