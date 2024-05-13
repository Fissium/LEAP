import torch
import torch.nn as nn


class WaveBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_rates: int, kernel_size: int
    ):
        super().__init__()
        self.num_rates = num_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2**i for i in range(num_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate * (kernel_size - 1)) / 2),
                    dilation=dilation_rate,
                )
            )
            self.gate_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate * (kernel_size - 1)) / 2),
                    dilation=dilation_rate,
                )
            )
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(
                self.gate_convs[i](x)
            )
            x = self.convs[i + 1](x)
            res = res + x
        return res


class WaveNet(nn.Module):
    def __init__(self, inch=9, kernel_size=3):
        super().__init__()
        self.wave_block1 = WaveBlock(inch, 16, 12, kernel_size)
        self.wave_block2 = WaveBlock(16, 32, 8, kernel_size)
        self.wave_block3 = WaveBlock(32, 64, 4, kernel_size)
        self.wave_block4 = WaveBlock(64, 128, 1, kernel_size)
        self.wave_block5 = WaveBlock(128, 6, 1, kernel_size)
        self.fc_in = nn.Linear(360 + 16 + 180, 256)
        self.fc_out = nn.Linear(256, 8)

    def forward(self, x_seq, x_scalar):
        x = self.wave_block1(x_seq)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        x = self.wave_block4(x)
        x = self.wave_block5(x)
        x = x.reshape(x.shape[0], -1)
        y = nn.functional.gelu(self.fc_in(torch.cat((x, x_scalar), dim=1)))
        y = self.fc_out(y)
        return torch.cat((x, y), dim=1)
