import torch
import torch.nn as nn


class WaveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super().__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2**i for i in range(dilation_rates)]
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

    def forward(self, x):
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
    def __init__(self, output_size, inch=9, kernel_size=3):
        super().__init__()
        self.wave_block1 = WaveBlock(inch, 16, 12, kernel_size)
        self.wave_block2 = WaveBlock(16, 32, 8, kernel_size)
        self.wave_block3 = WaveBlock(32, 64, 4, kernel_size)
        self.wave_block4 = WaveBlock(64, 128, 1, kernel_size)
        self.fc_out = nn.Linear(1024, output_size)
        self.fc_in = nn.Linear(60 * 128, 1024)

    def forward(self, x):
        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        x = self.wave_block4(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.functional.relu(self.fc_in(x))
        x = self.fc_out(x)
        return x


class WaveNetLSTM(nn.Module):
    def __init__(self, output_size, inch=9, kernel_size=3):
        super().__init__()
        self.wave_block1 = WaveBlock(inch, 16, 12, kernel_size)
        self.wave_block2 = WaveBlock(16, 32, 8, kernel_size)
        self.wave_block3 = WaveBlock(32, 64, 4, kernel_size)
        self.wave_block4 = WaveBlock(64, 128, 1, kernel_size)
        self.lstm = nn.LSTM(
            128 + 25,
            64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_in = nn.Linear(60 * 128, 1024)
        self.fc_out = nn.Linear(1024, output_size)

    def forward(self, x):
        x_inp = x
        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        x = self.wave_block4(x)
        x, _ = self.lstm(torch.cat((x.permute(0, 2, 1), x_inp.permute(0, 2, 1)), dim=2))
        x = x.reshape(x.shape[0], -1)
        x = nn.functional.relu(self.fc_in(x))
        x = self.fc_out(x)
        return x


class WaveNetLSTMAttention(nn.Module):
    def __init__(self, output_size, inch=9, kernel_size=3):
        super().__init__()
        self.wave_block1 = WaveBlock(inch, 16, 12, kernel_size)
        self.wave_block2 = WaveBlock(16, 32, 8, kernel_size)
        self.wave_block3 = WaveBlock(32, 64, 4, kernel_size)
        self.wave_block4 = WaveBlock(64, 128, 1, kernel_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=60,
                nhead=6,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.lstm = nn.LSTM(
            128 + 25,
            64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_in = nn.Linear(60 * 128, 1024)
        self.fc_out = nn.Linear(1024, output_size)

    def forward(self, x):
        x_wavent = self.wave_block1(x)
        x_wavent = self.wave_block2(x_wavent)
        x_wavent = self.wave_block3(x_wavent)
        x_wavent = self.wave_block4(x_wavent)

        x_transformer = self.transformer(x)
        y, _ = self.lstm(
            torch.cat(
                (
                    x_wavent.permute(0, 2, 1),
                    x_transformer.permute(0, 2, 1),
                ),
                dim=2,
            )
        )
        y = y.reshape(y.shape[0], -1)
        y = nn.functional.relu(self.fc_in(y))
        y = self.fc_out(y)
        return y
