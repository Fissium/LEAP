import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.scale = in_channels**-0.5

    def forward(self, x):
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        scores = torch.bmm(query.permute(0, 2, 1), key) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, value.permute(0, 2, 1)).permute(0, 2, 1)

        return context + x


class WaveNet(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int):
        super().__init__()
        self.model = nn.Sequential(
            WaveBlock(in_channels, 16, 12, kernel_size),
            nn.BatchNorm1d(16),
            AttentionBlock(16),
            WaveBlock(16, 32, 8, kernel_size),
            nn.BatchNorm1d(32),
            AttentionBlock(32),
            WaveBlock(32, 64, 4, kernel_size),
            nn.BatchNorm1d(64),
            AttentionBlock(64),
            WaveBlock(64, 128, 1, kernel_size),
            nn.BatchNorm1d(128),
            AttentionBlock(128),
            WaveBlock(128, 3 * 6 + 8, 1, kernel_size),
        )

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int):
        super().__init__()
        self.model = WaveNet(in_channels, kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.model(x)
        x_seq = x[:, :6, :].reshape(x.size(0), -1)
        x_delta_first = x[:, 6:12, :].reshape(x.size(0), -1)
        x_delta_second = x[:, 12:18, :].reshape(x.size(0), -1)

        x_scalar = self.global_avg_pool(x[:, 18:, :]).squeeze(dim=-1)

        x_seq = torch.cat((x_seq, x_scalar), dim=1)

        return x_seq, x_delta_first, x_delta_second
