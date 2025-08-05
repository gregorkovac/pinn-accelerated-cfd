import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels=2, num_layers=3, hidden_channels=64, input_width=20, input_height=20):
        super(CNN, self).__init__()
        
        self.input_width = input_width
        self.input_height = input_height

        encoder_layers = []

        for i in range(num_layers):
            in_channels = input_channels if i == 0 else hidden_channels

            encoder_layers.extend([
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="replicate"),
                nn.BatchNorm2d(hidden_channels),
                nn.Tanh()
            ])

        self.encoder = nn.Sequential(*encoder_layers)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1, padding_mode="replicate")
        )

        self.initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        x = x.squeeze(1)

        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)