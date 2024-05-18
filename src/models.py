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
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.wave_block2 = WaveBlock(16, 32, 8, kernel_size)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.wave_block3 = WaveBlock(32, 64, 4, kernel_size)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.wave_block4 = WaveBlock(64, 128, 1, kernel_size)
        self.batchnorm4 = nn.BatchNorm1d(128)
        self.wave_block_x = WaveBlock(128, 14, 1, kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.wave_block_delta_first = WaveBlock(128, 14, 1, kernel_size)
        self.wave_block_x_delta_second = WaveBlock(128, 14, 1, kernel_size)

    def forward(self, x):
        x = self.wave_block1(x)
        x = self.batchnorm1(x)
        x = self.wave_block2(x)
        x = self.batchnorm2(x)
        x = self.wave_block3(x)
        x = self.batchnorm3(x)
        x = self.wave_block4(x)
        x_shared = self.batchnorm4(x)
        x = self.wave_block_x(x_shared)
        x_delta_first = self.wave_block_delta_first(x_shared)
        x_delta_second = self.wave_block_x_delta_second(x_shared)

        x_delta_first = x_delta_first[:, :6, :].reshape(x_delta_first.size(0), -1)
        x_delta_second = x_delta_second[:, :6, :].reshape(x_delta_second.size(0), -1)

        x_seq = x[:, :6, :].reshape(x.size(0), -1)
        x_scalar = self.pool(x[:, 6:, :]).squeeze(dim=-1)

        x = torch.cat((x_seq, x_scalar), dim=1)

        return x, x_delta_first, x_delta_second
