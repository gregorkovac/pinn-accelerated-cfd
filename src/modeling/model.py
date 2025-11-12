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

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "tanh":
            activation_fn = nn.Tanh()

        if architecture == "cnn":
            pooling_fn = nn.Identity()
            upsampling_fn = nn.Identity()
            stride = 1
        elif architecture == "encdec":
            if pooling == "max":
                pooling_fn = nn.MaxPool2d(2)
                stride = 1
            elif pooling == "avg":
                pooling_fn = nn.AvgPool2d(2)
                stride = 1
            elif pooling == "stride":
                pooling_fn = nn.Identity()
                stride = 2

            if upsampling == "upsample":
                upsampling_fn = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            elif upsampling == "convtranspose":
                upsampling_fn = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2, padding=0, output_padding=0)

        if norm == "none":
            norm_fn = nn.Identity()
        elif norm == "batch":
            norm_fn = nn.BatchNorm2d(hidden_channels)
        elif norm == "group":
            norm_fn = nn.GroupNorm(1, hidden_channels)

        for i in range(num_layers):
            in_channels = input_channels if i == 0 else hidden_channels

            encoder_layers.extend([
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="zeros", stride=stride),
                norm_fn,
                activation_fn,
                pooling_fn
            ])

            if architecture == "encdec" and i < num_layers - 1:
                decoder_layers.extend([
                    upsampling_fn,
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode="zeros"),
                    activation_fn
                ])

        decoder_layers.extend([
            upsampling_fn,
            nn.Conv2d(hidden_channels, output_channels, 3, padding=1, padding_mode="zeros"),
        ])

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self.final_conv = nn.Conv2d(input_channels + output_channels, output_channels, 3, padding=1, padding_mode="zeros")

        if use_skip:
            self.skip = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        else:
            self.skip = None

        # self.initialize_weights()

    def forward(self, x):

        out = self.encoder(x)
        out = self.decoder(out)

        if self.skip is not None:
            # out = out + self.skip(x)
            x_skip = self.skip(x)
            out = torch.cat([out, x_skip], dim=1)
            out = self.final_conv(out)

        out = self.enforceBC(x, out)

        # out = out.squeeze(1)

        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def enforceBC(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return out