from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class PermuteLayer(torch.nn.Module):
    dims: tuple[int, ...]

    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.permute(*self.dims)


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class EmbeddingLayerFC(nn.Module):
    def __init__(
        self,
        in_chans: int = 19,
        embed_dim: int = 128,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.embeddings = nn.Linear(in_chans, embed_dim)
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.norm(x)
        return x


class EmbeddingLayerConv(nn.Module):
    def __init__(
        self,
        in_chans: int = 19,
        embed_dim: int = 128,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm1d,
    ):
        super().__init__()
        self.embeddings = nn.Conv1d(
            in_channels=in_chans, out_channels=embed_dim, kernel_size=3, padding=1
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x.permute(0, 2, 1))
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        L (int): Length of data

    Returns:
        x: (B, L, C)
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    x = x.contiguous().view(B, L, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(1, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # get relative_coords_table
        relative_coords_w = torch.arange(
            -(self.window_size - 1), self.window_size, dtype=torch.float32
        )
        relative_coords_table = relative_coords_w  # 2*W-1
        if pretrained_window_size > 0:
            relative_coords_table[:] /= pretrained_window_size - 1
        else:
            relative_coords_table[:] /= self.window_size - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / np.log2(8)
        )

        self.register_buffer(
            "relative_coords_table", relative_coords_table.unsqueeze(1)
        )  # (2*W-1, 1)

        # get pair-wise relative position index for each token inside the window
        coords_w = torch.arange(self.window_size)
        relative_coords = coords_w[:, None] - coords_w[None, :]  # W, W
        relative_coords[:, :] += self.window_size - 1  # shift to start from 0
        # relative_position_index | example
        # [2, 1, 0]
        # [3, 2, 1]
        # [4, 3, 2]
        self.register_buffer(
            "relative_position_index", relative_coords
        )  # (W, W): range of 0 -- 2*(W-1)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        device = next(self.parameters()).device
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (  # type: ignore
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),  # type: ignore
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01, device=device))
        ).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table
        )  # (2*W-1, nH)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size, self.window_size, -1)  # (W, W, nH)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, W, W
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=pretrained_window_size,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, L, C = x.shape
        assert (
            L >= self.window_size
        ), f"input length ({L}) must be >= window size ({self.window_size})"
        assert (
            L % self.window_size == 0
        ), f"input length ({L}) must be divisible by window size ({self.window_size})"

        shortcut = x

        # zero-padding shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            shifted_x[:, -self.shift_size :] = 0.0
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, C

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, L)  # (B, L, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
            x[:, : self.shift_size] = 0.0  # remove invalid embs
        else:
            x = shifted_x

        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class SwinTransformerV2Layer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class Model(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depths: list[int],
        in_chans: int,
        num_heads: list[int],
        window_size: int,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        max_len=60,
        embed_layer="fc",
        head="mlp",
        norm_layer=nn.LayerNorm,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_layers = len(depths)

        self.max_len = max_len

        if embed_layer == "fc":
            self.embeddings = EmbeddingLayerFC(
                in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
            )
        elif embed_layer == "conv":
            self.embeddings = EmbeddingLayerConv(in_chans=in_chans, embed_dim=embed_dim)
        else:
            raise NotImplementedError

        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinTransformerV2Layer(
                embed_dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                drop_path=drop_path,
                pretrained_window_size=pretrained_window_size,
            )
            self.layers.append(layer)

        if embed_layer == "fc":
            if head == "pool":
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(19), PermuteLayer((0, 2, 1))
                )
            elif head == "mlp":
                self.head = nn.Sequential(
                    Mlp(in_features=embed_dim, out_features=19), PermuteLayer((0, 2, 1))
                )
            else:
                raise NotImplementedError
        elif embed_layer == "conv":
            self.head = nn.Sequential(
                PermuteLayer((0, 2, 1)),
                nn.Conv1d(
                    in_channels=embed_dim, out_channels=19, kernel_size=3, padding=1
                ),
            )
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        trunc_normal_(self.pos_embed, std=0.02)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embeddings(x)
        x = x + self.pos_embed[:, : self.max_len, :]

        for layer in self.layers:
            x = layer(x)

        x = self.head(x)

        y_seq = x[:, :6, :].reshape(x.size(0), -1)
        y_delta_first = x[:, 6:12, :].reshape(x.size(0), -1)
        y_delta_second = x[:, 12:18, :].reshape(x.size(0), -1)

        # scalar outputs must be non-negative
        y_scalar = self.relu(x[:, 18:, :8].reshape(x.size(0), -1))

        y_seq = torch.cat([y_seq, y_scalar], dim=-1)

        return y_seq, y_delta_first, y_delta_second
