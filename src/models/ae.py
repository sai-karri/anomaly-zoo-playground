import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.base_channels = base_channels

        layers = []
        in_channels = 3
        for i in range(5):
            out_channels = self.base_channels * (2 ** i)

            layers.extend([
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=4,
                          stride=2,
                          padding=1
                          ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ])

            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.base_channels = base_channels

        layers = []
        in_channels = self.base_channels * 16
        for i in range(1,5):
            out_channels = in_channels // 2

            layers.extend([
                nn.ConvTranspose2d(
                          in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=4,
                          stride=2,
                          padding=1
                          ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ])

            in_channels = out_channels
        layers.extend([nn.ConvTranspose2d(
                          in_channels=in_channels,
                          out_channels=3,
                          kernel_size=4,
                          stride=2,
                          padding=1
                          )])
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, base_channels = 32):
        super().__init__()
        self.encoder = Encoder(base_channels)
        self.decoder = Decoder(base_channels)

    def forward(self, x):
        encode_out = self.encoder(x)

        return self.decoder(encode_out)


if __name__ == '__main__':
    model = AutoEncoder(base_channels=32)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(out.shape)