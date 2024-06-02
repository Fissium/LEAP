import math

import torch
import torch.nn as nn


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
        max_len: int = 5000,
    ):
        super().__init__()

        self.embedding = nn.Linear(in_features=in_channels, out_features=d_model)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            nheads=nheads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_len=max_len,
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(in_features=d_model, out_features=6 * 3 + 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)

        y_seq = x[:, :6, :].reshape(x.size(0), -1)
        y_delta_first = x[:, 6:12, :].reshape(x.size(0), -1)
        y_delta_second = x[:, 12:18, :].reshape(x.size(0), -1)

        y_scalar = self.global_avg_pool(x[:, 18:, :8]).reshape(x.size(0), -1)

        y_seq = torch.cat([y_seq, y_scalar], dim=-1)

        return y_seq, y_delta_first, y_delta_second
