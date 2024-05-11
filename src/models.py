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
        self.fc_in = nn.Linear(60 * 128 + 16, 1024)

    def forward(self, x):
        x_seq = torch.cat((x[:, :360], x[:, -180:]), dim=1).view(x.shape[0], 9, 60)
        x_scalar = x[:, 360:376]

        x = self.wave_block1(x_seq)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        x = self.wave_block4(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat((x, x_scalar), dim=1)
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
            128 + 60, 128, num_layers=2, batch_first=True, bidirectional=True
        )
        self.fc_in = nn.Linear(18 * 128 + 16, 1024)
        self.fc_out = nn.Linear(1024, output_size)
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)
            elif "fc" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "bias" in name:
                    p.data.fill_(0)

    def forward(self, x):
        x_seq = torch.cat((x[:, :360], x[:, -180:]), dim=1).view(x.shape[0], 60, 9)
        x_scalar = x[:, 360:376]

        x = self.wave_block1(x_seq)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        x = self.wave_block4(x)
        x, _ = self.lstm(torch.cat((x, x_seq), dim=1).permute(0, 2, 1))
        x = x.reshape(x.shape[0], -1)
        x = torch.cat((x, x_scalar), dim=1)
        x = nn.functional.relu(self.fc_in(x))
        x = self.fc_out(x)
        return x
