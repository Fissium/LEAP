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

        self.convs.append(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )
        dilation_rates = [2**i for i in range(num_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate * (kernel_size - 1)) / 2),
                    dilation=dilation_rate,
                )
            )
            self.gate_convs.append(
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate * (kernel_size - 1)) / 2),
                    dilation=dilation_rate,
                )
            )
            self.convs.append(
                nn.Conv1d(
                    in_channels=out_channels, out_channels=out_channels, kernel_size=1
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, in_channels: int, kernel_size: int):
        super().__init__()
        self.model = nn.Sequential(
            WaveBlock(
                in_channels=in_channels,
                out_channels=16,
                num_rates=12,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm1d(num_features=16),
            WaveBlock(
                in_channels=16, out_channels=32, num_rates=8, kernel_size=kernel_size
            ),
            nn.BatchNorm1d(num_features=32),
            WaveBlock(
                in_channels=32, out_channels=64, num_rates=4, kernel_size=kernel_size
            ),
            nn.BatchNorm1d(num_features=64),
            WaveBlock(
                in_channels=64, out_channels=128, num_rates=1, kernel_size=kernel_size
            ),
            nn.BatchNorm1d(num_features=128),
            WaveBlock(
                in_channels=128,
                out_channels=3 * 6 + 1,
                num_rates=1,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int):
        super().__init__()
        self.model = WaveNet(in_channels=in_channels, kernel_size=kernel_size)
        # self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.model(x)
        y_seq = x[:, :6, :].reshape(x.size(0), -1)
        y_delta_first = x[:, 6:12, :].reshape(x.size(0), -1)
        y_delta_second = x[:, 12:18, :].reshape(x.size(0), -1)

        # y_scalar = self.global_avg_pool(x[:, 18:, :]).squeeze(dim=-1)
        y_scalar = x[:, 18:, :8].reshape(x.size(0), -1)

        y_seq = torch.cat((y_seq, y_scalar), dim=1)

        return y_seq, y_delta_first, y_delta_second
