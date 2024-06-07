import math

import rootutils
import torch
import torch.nn as nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.wavenet import WaveBlock  # noqa: E402


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # do not train

        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.encoding[:, :length, :].to(x.device)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        nheads: int,
        forward_expansion: int,
        dropout: float,
        max_len: int = 5000,
    ):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nheads,
                dim_feedforward=forward_expansion * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.positional_encoding(x)
        out = self.layers(x, src_key_padding_mask=mask)
        return out


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 19,
        d_model: int = 192,
        num_layers: int = 4,
        nheads: int = 4,
        forward_expansion: int = 2,
        dropout: float = 0.0,
        kernel_size: int = 3,
        max_len: int = 5000,
    ):
        super().__init__()

        self.embedding = nn.Linear(in_features=in_channels, out_features=d_model // 2)
        self.wavenet = nn.Sequential(
            WaveBlock(
                in_channels=in_channels,
                out_channels=16,
                num_rates=12,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm1d(num_features=16),
            WaveBlock(
                in_channels=16,
                out_channels=32,
                num_rates=8,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm1d(num_features=32),
            WaveBlock(
                in_channels=32,
                out_channels=64,
                num_rates=4,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm1d(num_features=64),
            WaveBlock(
                in_channels=64,
                out_channels=d_model // 2,
                num_rates=1,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm1d(num_features=d_model // 2),
        )

        self.layer_norm = nn.LayerNorm(d_model // 2)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            nheads=nheads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_len=max_len,
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=19)

    def forward(self, x_inp: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x_wavenet = self.wavenet(x_inp).permute(0, 2, 1)
        x = self.embedding(x_inp.permute(0, 2, 1))
        x = self.layer_norm(x)
        x = torch.cat([x, x_wavenet], dim=-1)
        x = self.transformer(x)
        x = self.global_avg_pool(x)
        x = x.permute(0, 2, 1)

        y_seq = x[:, :6, :].reshape(x.size(0), -1)
        y_delta_first = x[:, 6:12, :].reshape(x.size(0), -1)
        y_delta_second = x[:, 12:18, :].reshape(x.size(0), -1)

        y_scalar = x[:, 18:, :8].reshape(x.size(0), -1)

        y_seq = torch.cat([y_seq, y_scalar], dim=-1)

        return y_seq, y_delta_first, y_delta_second
