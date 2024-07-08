from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


def init_weights_(module: nn.Module, name: str = ""):  # noqa: ARG001
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def drop_add_residual_stochastic_depth(
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
) -> torch.Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * inputs.sigmoid()


class DoubleLevelWiseConv(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, level_size: int = 3):
        super().__init__()

        self.bn_00 = nn.BatchNorm2d(in_chans)
        self.depthwise_01 = nn.Conv2d(
            in_channels=in_chans,
            out_channels=out_chans,
            kernel_size=(level_size, 1),
            stride=1,
            padding=((level_size - 1) // 2, 0),
        )
        self.silu = Swish()
        self.bn_01 = nn.BatchNorm2d(out_chans)
        self.depthwise_02 = nn.Conv2d(
            in_channels=out_chans,
            out_channels=out_chans,
            kernel_size=(level_size, 1),
            stride=1,
            padding=((level_size - 1) // 2, 0),
        )
        self.silu = Swish()
        self.bn_02 = nn.BatchNorm2d(out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_00(x)
        x = self.depthwise_01(x)
        x = self.silu(x)
        x = self.bn_01(x)
        x = self.depthwise_02(x)
        x = self.silu(x)
        x = self.bn_02(x)
        return x


class ChannelMixing(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()

        self.first_linear = nn.Linear(in_chans, out_chans)
        self.gelu = Swish()
        self.layer_norm = nn.LayerNorm(out_chans)
        self.second_linear = nn.Linear(out_chans, in_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.first_linear(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.second_linear(x)
        return x + residual


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
        out_chans: int = 64,
    ):
        super().__init__()
        self.level_wise_conv_01 = DoubleLevelWiseConv(
            in_chans=in_chans, out_chans=out_chans, level_size=3
        )
        self.level_wise_conv_02 = DoubleLevelWiseConv(
            in_chans=in_chans, out_chans=out_chans, level_size=7
        )
        self.level_wise_conv_03 = DoubleLevelWiseConv(
            in_chans=in_chans, out_chans=out_chans, level_size=15
        )
        self.channel_mixer_01 = ChannelMixing(
            in_chans=out_chans, out_chans=out_chans * 2
        )
        self.channel_mixer_02 = ChannelMixing(
            in_chans=out_chans, out_chans=out_chans * 2
        )
        self.channel_mixer_03 = ChannelMixing(
            in_chans=out_chans, out_chans=out_chans * 2
        )
        self.level_wise_conv_04 = DoubleLevelWiseConv(
            in_chans=out_chans * 3, out_chans=out_chans * 3, level_size=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 3, 1)
        level_conved_01 = self.level_wise_conv_01(x).squeeze(3)
        level_conved_02 = self.level_wise_conv_02(x).squeeze(3)
        level_conved_03 = self.level_wise_conv_03(x).squeeze(3)

        level_conved_01 = level_conved_01.permute(0, 2, 1)
        level_conved_02 = level_conved_02.permute(0, 2, 1)
        level_conved_03 = level_conved_03.permute(0, 2, 1)
        mixed_01 = self.channel_mixer_01(level_conved_01)
        mixed_02 = self.channel_mixer_02(level_conved_02)
        mixed_03 = self.channel_mixer_03(level_conved_03)

        x = torch.cat([mixed_01, mixed_02, mixed_03], dim=2)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float | torch.Tensor = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] | None = None,  # noqa: ARG002
        drop: float = 0.0,  # noqa: ARG002
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


class Model(nn.Module):
    def __init__(
        self,
        in_chans: int = 19,
        out_chans: int = 13,
        max_len: int = 5000,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: float | None = None,  # for layerscale: None or 0 => no layerscale
        embed_layer: str = "fc",
        head: str = "pool",
        act_layer: Callable[..., nn.Module] = nn.GELU,
        block_fn: Callable[..., nn.Module] = Block,
        ffn_layer="mlp",
        block_chunks: int = 1,
    ):
        super().__init__()
        self.max_len = max_len
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if embed_layer == "fc":
            self.embeddings = EmbeddingLayerFC(
                in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
            )
        elif embed_layer == "conv":
            self.embeddings = EmbeddingLayerConv(
                in_chans=in_chans, out_chans=embed_dim // 3
            )
        else:
            raise NotImplementedError

        self.n_blocks = depth
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer = Mlp

        elif ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFN
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        if head == "pool":
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(out_chans), PermuteLayer((0, 2, 1))
            )
        elif head == "mlp":
            self.head = nn.Sequential(
                Mlp(in_features=embed_dim, out_features=out_chans),
                PermuteLayer((0, 2, 1)),
            )
        else:
            raise NotImplementedError

        self.relu = nn.ReLU()
        self.alphas = nn.Parameter(torch.zeros(self.n_blocks))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_, self)

    def forward(self, x):
        x = self.embeddings(x)
        x = x + self.pos_embed[:, : self.max_len, :]

        for idx, blk in enumerate(self.blocks):
            x_old = x
            x = blk(x)
            x = x + self.alphas[idx] * x_old

        x = self.head(x)

        y = x[:, :6, :].reshape(x.size(0), -1)
        y_delta = x[:, 6:12, :].reshape(x.size(0), -1)

        # scalar outputs must be non-negative
        y_scalar = self.relu(x[:, 12:, :8].reshape(x.size(0), -1))

        y = torch.cat([y, y_scalar], dim=-1)

        return y, y_delta
