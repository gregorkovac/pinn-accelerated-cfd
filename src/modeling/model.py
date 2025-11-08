import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, 
                 input_channels=2, 
                 output_channels=1, 
                 num_layers=3, 
                 hidden_channels=64, 
                 input_width=20, 
                 input_height=20, 
                 enforceBC=lambda x, out: out,
                 use_skip=False):
        
        super(CNN, self).__init__()
        
        self.input_width = input_width
        self.input_height = input_height

        self.output_channels = output_channels

        self.enforceBC = enforceBC

        encoder_layers = []
        decoder_layers = []

        for i in range(num_layers):
            in_channels = input_channels if i == 0 else hidden_channels

            encoder_layers.extend([
                # nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="zeros"),
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="replicate", stride=2),
                # nn.GroupNorm(1, hidden_channels),
                # nn.BatchNorm2d(hidden_channels),
                # nn.LeakyReLU(),
                nn.Tanh(),
                # nn.AvgPool2d(2)
                # nn.MaxPool2d(2)
                # nn.LayerNorm([hidden_channels, input_height, input_width]),
                # nn.SiLU()
            ])

            if i < num_layers - 1:
                decoder_layers.extend([
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode="replicate"),
                    # nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
                    nn.Tanh()
                    # nn.LeakyReLU()
                    # nn.SiLU()
                ])

        decoder_layers.extend([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, output_channels, 3, padding=1, padding_mode="replicate"),
            # nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            # nn.Tanh()
            # nn.ReLU()
        ])

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self.final_conv = nn.Conv2d(input_channels + output_channels, output_channels, 3, padding=1, padding_mode="replicate")

        # self.decoder = nn.Sequential(
        #     # nn.Conv2d(hidden_channels, output_channels, 3, padding=1, padding_mode="zeros")
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(hidden_channels, output_channels, 3, padding=1, padding_mode="zeros")
        # )

        if use_skip:
            self.skip = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        else:
            self.skip = None

        # self.initialize_weights()

    def forward(self, x):

        out = self.encoder(x)
        out = self.decoder(out)

        if self.skip is not None:
            # out = out + self.skip(x)
            # x_skip = self.skip(x)
            out = torch.cat([out, x], dim=1)
            out = self.final_conv(out)

        out = self.enforceBC(x, out)

        out = out.squeeze(1)

        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def enforceBC(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return out