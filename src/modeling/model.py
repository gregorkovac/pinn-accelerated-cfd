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
                 use_skip=False,
                 activation="relu",
                 architecture="cnn",
                 pooling="avg",
                 upsampling="upsample",
                 norm="none"):
        
        super(CNN, self).__init__()
        
        self.input_width = input_width
        self.input_height = input_height

        self.output_channels = output_channels

        self.enforceBC = enforceBC

        encoder_layers = []
        decoder_layers = []

        for i in range(num_layers):
            in_channels = input_channels if i == 0 else hidden_channels
        
            if activation == "relu":
                activation_fn = nn.ReLU()
            elif activation == "tanh":
                activation_fn = nn.Tanh()

            if norm == "none":
                norm_fn = nn.Identity()
            elif norm == "batch":
                norm_fn = nn.BatchNorm2d(hidden_channels)
            elif norm == "group":
                norm_fn = nn.GroupNorm(1, hidden_channels)

            encoder_layers.extend([
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="zeros", stride=1),
                norm_fn,
                activation_fn,
            ])

        decoder_layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="zeros", stride=1),
        ])

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        if use_skip:
            self.skip = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
            self.final_conv = nn.Conv2d(hidden_channels * 2, output_channels, kernel_size=1)
        else:
            self.skip = None
            self.final_conv = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x):
        out = self.encoder(x)


        out = self.decoder(out)

        if self.skip is not None:
            x_skip = self.skip(x)
            out = torch.cat([out, x_skip], dim=1)

        out = self.final_conv(out)
        

        out = self.enforceBC(x, out)

        return out
    
    def initialize_weights(self, activation_fn="tanh"):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    if activation_fn == "tanh":
                        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                    elif activation_fn == "relu":
                        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def enforceBC(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return out