import math

import rootutils
import torch
import torch.nn as nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# class SequenceToScalarTransformer(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         d_model,
#         nhead,
#         num_encoder_layers,
#         dim_feedforward,
#         dropout=0.1,
#     ):
#         super().__init__()
#
#         # Linear layer to project the input features to the model dimension
#         self.input_linear = nn.Linear(input_dim, d_model)
#
#         # Transformer Encoder to process the sequence
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=d_model,
#                 nhead=nhead,
#                 dim_feedforward=dim_feedforward,
#                 dropout=dropout,
#             ),
#             num_layers=num_encoder_layers,
#         )
#
#         self.output_linear = nn.Linear(d_model, output_dim)
#
#     def forward(self, src):
#         # src shape: [seq_len, batch_size, input_dim]
#
#         # Project input to model dimension
#         src = self.input_linear(src).permute(1, 0, 2)
#
#         # Process sequence with the transformer encoder
#         src = self.transformer_encoder(src)  # shape: [seq_len, batch_size, d_model]
#
#         # Pooling over the sequence dimension, aggregate information
#         src = src.mean(dim=0)  # shape: [batch_size, d_model]
#
#         # Map to the desired output dimension
#         output = self.output_linear(src)  # shape: [batch_size, output_dim]
#
#         return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, max_len: int = 5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        self.encoding.requires_grad = False  # we don't want to train this

        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size)
        )

        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        length = x.size(1)
        return x + self.encoding[:, :length, :].to(x.device)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, self.embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_size: int, num_heads: int, dropout: float, forward_expansion: int
    ):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_layers: int,
        num_heads: int,
        forward_expansion: int,
        dropout: float,
        max_len: int = 5000,
    ):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        out = self.positional_encoding(x)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class TransformerModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 25,
        embed_size: int = 192,
        num_layers: int = 6,
        num_heads: int = 6,
        forward_expansion: int = 1,
        dropout: float = 0.0,
        max_len: int = 10_000,
    ):
        super().__init__()
        self.embedding = nn.Linear(in_channels, embed_size)
        self.transformer = TransformerEncoder(
            embed_size, num_layers, num_heads, forward_expansion, dropout, max_len
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(26)

        self.global_avg_pool_scalar = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x.permute(0, 2, 1))
        x = self.transformer(x, mask=None)
        x = self.global_avg_pool(x)
        x = x.permute(0, 2, 1)

        x_seq = x[:, :6, :].reshape(x.size(0), -1)
        x_delta_first = x[:, 6:12, :].reshape(x.size(0), -1)
        x_delta_second = x[:, 12:18, :].reshape(x.size(0), -1)

        x_scalar = self.global_avg_pool_scalar(x[:, 18:, :]).squeeze(dim=-1)

        x_seq = torch.cat([x_seq, x_scalar], dim=-1)

        return x_seq, x_delta_first, x_delta_second
