import torch
import torch.nn as nn

class EncDecCNN(nn.Module):
    def __init__(self, 
                 input_channels=2, 
                 output_channels=1, 
                 num_layers=3, 
                 hidden_channels=64, 
                 input_width=20, 
                 input_height=20, 
                 enforceBC=lambda x, out: out,
                 skips="none",
                 activation="relu",
                 pooling="avg",
                 upsampling="upsample",
                 norm="none"):
        
        super(EncDecCNN, self).__init__()
        
        self.input_width = input_width
        self.input_height = input_height

        self.output_channels = output_channels

        self.enforceBC = enforceBC

        self.skips = skips

        self.encoder = []
        self.decoder = []

        # if activation == "relu":
        #     activation_fn = nn.ReLU()
        # elif activation == "tanh":
        #     activation_fn = nn.Tanh()
        # elif activation == "silu":
        #     activation_fn = nn.SiLU()

        # if norm == "none":
        #     norm_fn = nn.Identity()
        # elif norm == "batch":
        #     norm_fn = nn.BatchNorm2d(out_channels)
        # elif norm == "group":
        #     norm_fn = nn.GroupNorm(1, out_channels)
            
        # self.first_conv = nn.Sequential(
        #     nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="zeros"),
        #     norm_fn,
        
        #     activation_fn,
        # )

        # ENCODER
        for i in range(num_layers):
            # in_channels = input_channels * (2 ** i)
            # out_channels = input_channels * (2 ** (i + 1))

            if i == 0:
                in_channels = input_channels
            else:
                in_channels = hidden_channels * (2 ** i)

            out_channels = hidden_channels * (2 ** (i + 1))

            if activation == "relu":
                activation_fn = nn.ReLU()
            elif activation == "tanh":
                activation_fn = nn.Tanh()
            elif activation == "silu":
                activation_fn = nn.SiLU()

            # if i < 3:
            #     pooling_fn = nn.Identity()
            #     stride = 1
            if pooling == "max":
                pooling_fn = nn.MaxPool2d(2)
                stride = 1
            elif pooling == "avg":
                pooling_fn = nn.AvgPool2d(2)
                stride = 1
            elif pooling == "stride":
                pooling_fn = nn.Identity()
                stride = 2

            if norm == "none":
                norm_fn = nn.Identity()
            elif norm == "batch":
                norm_fn = nn.BatchNorm2d(out_channels)
            elif norm == "group":
                norm_fn = nn.GroupNorm(1, out_channels)
            elif norm == "layer":
                norm_fn = nn.LayerNorm(out_channels)

            # encoder_layers.extend([
            #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate", stride=stride),
            #     norm_fn,
            #     activation_fn,
            #     pooling_fn
            # ])

            encoder_block = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
                norm_fn,
                activation_fn,
                pooling_fn
            ]

            self.encoder.append(nn.Sequential(*encoder_block))

        #     print(in_channels, out_channels)

        # print("----------")

        # DECODER
        for i in range(num_layers):
            # if i == num_layers:
            #     out_channels = input_channels
            # else:

            # out_channels = hidden_channels
            # in_channels = hidden_channels
            
            # in_channels = input_channels * (2 ** (num_layers - i))
            # out_channels = input_channels * (2 ** (num_layers - i - 1))

            in_channels = hidden_channels * (2 ** (num_layers - i))
            out_channels = hidden_channels * (2 ** (num_layers - i - 1))

            if skips == "unet" and i != 0:
                in_channels *= 2

            # print(in_channels, out_channels)

            # if use_skip:
            #     in_channels = 2 * in_channels

            if activation == "relu":
                activation_fn = nn.ReLU()
            elif activation == "tanh":
                activation_fn = nn.Tanh()
            elif activation == "silu":
                activation_fn = nn.SiLU()

            # if i >= num_layers - 3:
            #     upsampling_fn = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="zeros")
            if upsampling == "upsample":
                upsampling_fn = nn.Sequential(
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                    # nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                                )
            elif upsampling == "convtranspose":
                upsampling_fn = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)


            # decoder_layers.extend([
            #     upsampling_fn,
            #     activation_fn
            # ])

            decoder_block = [
                upsampling_fn,
                activation_fn
            ]

            self.decoder.append(nn.Sequential(*decoder_block))

        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

        # self.encoder = nn.ModuleList(*encoder_layers)
        # self.decoder = nn.ModuleList(*decoder_layers)

        # fc_in = int((input_height / 2 ** (num_layers - 3)) * input_width / 2 ** (num_layers - 3) * input_channels * (2 ** num_layers))

        # self.fc = nn.Sequential(nn.Linear(fc_in, 128),
        #                         nn.ReLU(),
        #                         nn.Linear(128, fc_in),
        #                         nn.ReLU())

        # self.encoder = nn.ModuleList(encoder_layers)
        # self.decoder = nn.ModuleList(decoder_layers)

        # self.final_conv = nn.Conv2d(input_channels, output_channels, 3, padding=1, padding_mode="replicate")

        if skips == "single" or skips == "unet":
            self.global_skip = nn.Conv2d(input_channels, input_channels, kernel_size=1)
            self.final_conv = nn.Conv2d(input_channels + hidden_channels, output_channels, kernel_size=1)
            # self.final_conv = nn.Conv2d(2 * input_channels, output_channels, kernel_size=1)
        else:
            self.global_skip = None
            # self.final_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
            self.final_conv = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

        # if skips == "single":
        #     self.global_skip = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        #     self.final_conv = nn.Conv2d(2 * hidden_channels, output_channels, kernel_size=1)
        # elif skips == "unet":
        #     self.global_skip = None
        #     self.final_conv = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)
        # else:
        #     self.global_skip = None
        #     self.final_conv = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

        # self.initialize_weights(activation_fn)


    def forward(self, x):

        # x = self.first_conv(x)

        if self.skips == "unet":
            out_enc = [x]

            for layer in self.encoder:
                out_e = layer(out_enc[-1])
                out_enc.append(out_e)

            out = out_enc[-1]

            for i, layer in enumerate(self.decoder):
                if i != 0:
                    out = torch.cat([out, out_enc[-(i + 1)]], dim=1)

                out = layer(out)

            if self.global_skip is not None:
                x_skip = self.global_skip(x)
                out = torch.cat([out, x_skip], dim=1)
        else:
            out = self.encoder(x)
            out = self.decoder(out)

            if self.global_skip is not None:
                # out = out + self.skip(x)
                x_skip = self.global_skip(x)

                out = torch.cat([out, x_skip], dim=1)
        
        #     print(out_e.shape)

        # print("------------------")

        # out = self.encoder(x)

        # batch_size, channels, H, W = out.shape

        # out = out.view(batch_size, -1)
        # out = self.fc(out)
        # out = out.view(batch_size, channels, H, W)

        # out = self.decoder(out)

        # if self.skip is not None:
        #     # out = out + self.skip(x)
        #     x_skip = self.skip(x)

        #     out = torch.cat([out, x_skip], dim=1)
            # out = self.final_conv(out)

        # skip_connections = []
        # out = x

        # # ENCODER with storage
        # for block in self.encoder:
        #     out = block(out)
        #     if self.skip is not None:
        #         skip_connections.append(out)

        # # DECODER with concatenation
        # for i, block in enumerate(self.decoder):
        #     if self.skip is not None and len(skip_connections) > 0:
        #         # Take the last stored encoder feature
        #         skip_feat = skip_connections.pop()
        #         # Only skip if dimensions match (handling the bottleneck)
        #         if out.shape[2:] == skip_feat.shape[2:]:
        #             out = torch.cat([out, skip_feat], dim=1)
            
        #     out = block(out)

        out = self.final_conv(out)

        out = self.enforceBC(x, out)

        # # out = out.squeeze(1)

        return out
    
    def initialize_weights(self, activation_fn="tanh"):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    # nn.init.replicate_(m.weight)
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        # nn.init.replicate_(m.bias)
                        nn.init.constant_(m.bias, 0)
                else:
                    if activation_fn == "tanh":
                        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                    elif activation_fn == "relu":
                        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        # nn.init.replicate_(m.bias)
                        nn.init.constant_(m.bias, 0)

    def enforceBC(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return out