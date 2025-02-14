# This code was adapted from: https://github.com/openai/guided-diffusion/tree/main


import os
from abc import abstractmethod
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim

from models.basic_trainer import BasicTrainer

from models.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=.0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            # print(h.shape)
            hs.append(h)
        h = self.middle_block(h, emb)
        # print(h.shape)
        # print('n')
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        self.gap = nn.AvgPool2d((8, 8))  # global average pooling
        self.cam_feature_maps = None
        print('pool', pool)
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            print('spatial')
            #  self.out = nn.Linear(self._feature_size, self.out_channels)
            self.out = nn.Linear(256, self.out_channels)
            # nn.Sequential(
            #  nn.Linear(self._feature_size, 2048),
            #   nn.ReLU(),
            #    nn.Linear(self._feature_size, self.out_channels),
        # )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)

        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = torch.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)


def generate_simplex_noise(Simplex_instance, x, t, random_param=False, octave=6, persistence=0.8, frequency=64,
                           in_channels=1):
    noise = torch.empty(x.shape).to(x.device)
    for i in range(in_channels):
        Simplex_instance.newSeed()
        if random_param:
            import random
            param = random.choice(
                    [(2, 0.6, 16), (6, 0.6, 32), (7, 0.7, 32), (10, 0.8, 64), (5, 0.8, 16), (4, 0.6, 16), (1, 0.6, 64),
                     (7, 0.8, 128), (6, 0.9, 64), (2, 0.85, 128), (2, 0.85, 64), (2, 0.85, 32), (2, 0.85, 16),
                     (2, 0.85, 8),
                     (2, 0.85, 4), (2, 0.85, 2), (1, 0.85, 128), (1, 0.85, 64), (1, 0.85, 32), (1, 0.85, 16),
                     (1, 0.85, 8),
                     (1, 0.85, 4), (1, 0.85, 2), ]
                    )
            # 2D octaves seem to introduce directional artifacts in the top left
            noise[:, i, ...] = torch.unsqueeze(
                    torch.from_numpy(
                            # Simplex_instance.rand_2d_octaves(
                            #         x.shape[-2:], param[0], param[1],
                            #         param[2]
                            #         )
                            Simplex_instance.rand_3d_fixed_T_octaves(
                                    x.shape[-2:], t.detach().cpu().numpy(), param[0], param[1],
                                    param[2]
                                    )
                            ).to(x.device), 0
                    ).repeat(x.shape[0], 1, 1, 1)
        noise[:, i, ...] = torch.unsqueeze(
                torch.from_numpy(
                        # Simplex_instance.rand_2d_octaves(
                        #         x.shape[-2:], octave,
                        #         persistence, frequency
                        #         )
                        Simplex_instance.rand_3d_fixed_T_octaves(
                                x.shape[-2:], t.detach().cpu().numpy(), octave,
                                persistence, frequency
                                )
                        ).to(x.device), 0
                ).repeat(x.shape[0], 1, 1, 1)
    return noise


# covmodel_type = 0 %0=exponential; 1=expcos; 2=ebessel
def pcmc_atm(rows, cols, maxvar, alpha, covmodel_type, N, psizex, psizey, device='cuda'):
    n = rows * cols
    #     calculate distance between points in a matrix
    yy, xx = torch.meshgrid(torch.arange(0, rows), torch.arange(0, cols), indexing='ij')
    xxv = xx.reshape(n, 1).to(device)
    yyv = yy.reshape(n, 1).to(device)
    xxv = xxv * psizex
    yyv = yyv * psizey
    dx = xxv.repeat(1, n) - xxv.T.repeat(n, 1)
    dy = yyv.repeat(1, n) - yyv.T.repeat(n, 1)
    rgrid = torch.sqrt(dx ** 2 + dy ** 2).to(device)

    if covmodel_type == 0:
        #       Use exp function to calculate vcm
        vcm = maxvar * torch.exp(-alpha * rgrid)
    #     elif covmodel_type == 1:
    #         #       Use expcos function to calculate vcm
    #         vcm = maxvar * torch.exp(-alpha * rgrid) * torch.cos(beta * rgrid)
    # elif covmodel_type == 2:
    #     vcm = ebessel(rgrid, maxvar, eb_r, eb_w)

    #     Calculate correlated noise using Cholesky Decomposition
    #     1. create matrix of N gaussian noise vectors length n (on it's side!)
    Z = torch.randn(N, n).to(device)
    #     2. chol decomp on vcm
    V = torch.linalg.cholesky(vcm)  # upper triangular part of cholesky [V'V=vcm]

    #     3. Create matrix X containing N correlated noisevecotrs length n (on it's side!)
    X = torch.matmul(Z, V)
    atm_pets = X.view(N, rows, cols)

    return atm_pets


def atm_noise(x):
    max_vars = [5.5, 6.5, 7.5,  8.25, 9.]
    alphas = [0.004, 0.006, 0.008, 0.012, 0.016]
    ind = torch.randint(low=0, high=len(max_vars), size=(2,))

    noise = pcmc_atm(100, 100, max_vars[ind[0]], alphas[ind[1]], 0, x.shape[0], 1, 1)

    noise[noise > 30] = 30
    noise[noise < -30] = -30
    noise = (noise + 30) / 30 - 1

    transformer = torchvision.transforms.Resize((x.shape[-2], x.shape[-1]))
    noise = transformer(noise)
    noise = noise.view(x.shape)

    return noise


class PcmcAtm:
    def __init__(self, rows=100, cols=100, psizex=1, psizey=1, device='cuda'):
        n = rows * cols
        yy, xx = torch.meshgrid(torch.arange(0, rows), torch.arange(0, cols), indexing='ij')
        xxv = xx.reshape(n, 1).to(device)
        yyv = yy.reshape(n, 1).to(device)
        xxv = xxv * psizex
        yyv = yyv * psizey
        dx = xxv.repeat(1, n) - xxv.T.repeat(n, 1)
        dy = yyv.repeat(1, n) - yyv.T.repeat(n, 1)
        rgrid = torch.sqrt(dx ** 2 + dy ** 2).to(device)

        self.rgrid = rgrid
        self.n = n
        self.rows = rows
        self.cols = cols

        self.device = device

        self.max_vars = [5.5, 6.5, 7.5, 8.25, 9.]
        self.alphas = [0.004, 0.006, 0.008, 0.012, 0.016]

    def generate_atm(self, maxvar, alpha, N):
        vcm = maxvar * torch.exp(-alpha * self.rgrid)
        Z = torch.randn(N, self.n).to(self.device)
        V = torch.linalg.cholesky(vcm)
        X = torch.matmul(Z, V)
        atm_pets = X.view(N, self.rows, self.cols)

        return atm_pets

    def generate_noise(self, x):
        ind = torch.randint(low=0, high=5, size=(2,))

        noise = self.generate_atm(self.max_vars[ind[0]], self.alphas[ind[1]], x.shape[0])

        noise[noise > 30] = 30
        noise[noise < -30] = -30
        noise = (noise + 30) / 30 - 1

        transformer = torchvision.transforms.Resize((x.shape[-2], x.shape[-1]))
        noise = transformer(noise)
        noise = noise.view(x.shape)

        return noise


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        # so that I can index arrays with t=1...T, since x_0 = x so I ignore t=0
        self.noise_steps = noise_steps + 1
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.pcmc_atm = PcmcAtm(device=device)

        # self.noise_function = torch.randn_like
        self.noise_function = self.pcmc_atm.generate_noise

        self.beta = self.prepare_noise_schedule().to(device)
        self.beta = self.beta.view(-1, 1, 1, 1)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        # self.alpha_hat[0] = 1
        self.beta_hat = (1 - self.alpha_hat / self.alpha) / (1 - self.alpha_hat) * self.beta

        # self.simplex = SimplexCLASS()

        self.a = 1 / torch.sqrt(self.alpha)
        self.b = (1 - self.alpha) / torch.sqrt(1 - self.alpha_hat)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        # eps = generate_simplex_noise(self.simplex, x, t)
        eps = self.noise_function(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def remove_noise(self, model, x_t, t):
        pred = model(x_t, t)

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])

        return 1 / sqrt_alpha_hat * (x_t - sqrt_one_minus_alpha_hat * pred)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def reconstruct(self, model, x_t, t, label):
        pred = model(x_t, t, label)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        x = 1 / torch.sqrt(alpha) * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred)
        # x = (x_t - torch.sqrt(1 - alpha_hat) * pred) / torch.sqrt(alpha_hat)
        return x

    # def ddpm_sample(self, model, x_t, t):
    #     pred = model(x_t, t)
    #     alpha_hat = self.alpha_hat[t][:, None, None, None]
    #     alpha_hat_prev = self.alpha_hat[t-1][:, None, None, None]
    #     first = (x_t - torch.sqrt(1 - alpha_hat) * pred) / torch.sqrt(alpha_hat)
    #     third = torch.sqrt((1-alpha_hat_prev)/alpha_hat_prev)
    #     second = torch.sqrt(1 - alpha_hat_prev - torch.square(third)) * pred
    #     noise = torch.randn_like(x_t)
    #     return torch.sqrt(alpha_hat_prev) * first + second + third * noise
    def ddpm_sample(self, model, x_t, t, y=None):
        pred = model(x_t, t, y)
        # pred = pred[:, 1, :].view(-1, 1, 128, 128)
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        alpha_hat_prev = self.alpha_hat[t - 1][:, None, None, None]
        first = (x_t - torch.sqrt(1 - alpha_hat) * pred) / torch.sqrt(alpha_hat)
        second = torch.sqrt(1 - alpha_hat_prev) * pred
        return torch.sqrt(alpha_hat_prev) * first + second

    def ddim_sample(self, model, x_t, t, y, scaled_grad):
        pred = model(x_t, t, y)
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        alpha_hat_prev = self.alpha_hat[t-1][:, None, None, None]
        pred_hat = pred - torch.sqrt(1 - alpha_hat) * scaled_grad
        first = (x_t - torch.sqrt(1 - alpha_hat) * pred_hat) / torch.sqrt(alpha_hat)
        second = torch.sqrt(1 - alpha_hat_prev) * pred_hat
        return torch.sqrt(alpha_hat_prev) * first + second

    def ddim_reverse_sample(self, model, x_t, t, y=None):
        pred = model(x_t, t, y)
        # pred = pred[:, 1, :].view(-1, 1, 128, 128)
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        alpha_hat_next = self.alpha_hat[t+1][:, None, None, None]
        first = (torch.sqrt(1 / alpha_hat) - torch.sqrt(1 / alpha_hat_next)) * x_t
        second = (torch.sqrt(1 / alpha_hat_next - 1) - torch.sqrt(1 / alpha_hat - 1)) * pred
        return x_t + torch.sqrt(alpha_hat_next) * (first + second)

    def reverse_noising(self, model, x_t, t):
        pred = model(x_t, t)

        if t[0] > 1:
            # z = generate_simplex_noise(self.simplex, x_t, t)
            z = self.noise_function(x_t)
        else:
            z = 0

        # alpha = self.alpha[t].view(x_t.shape[0], 1, 1, 1)
        # alpha_hat = self.alpha_hat[t].view(x_t.shape[0], 1, 1, 1)
        a = self.a[t]
        b = self.b[t]
        beta_hat = self.beta_hat[t]
        beta = self.beta[t]

        return a * (x_t - b * pred) + torch.sqrt(beta) * z


class Trainer(BasicTrainer):
    def __init__(self, args, train_loader, test_loader, filename_sanitizer=(lambda x: x), logger=None):
        super().__init__(args, train_loader, test_loader, logger)
        # MODEL_FLAGS = "--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 " \
        #               "--num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
        attention_resolutions = "16"
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(self.image_size // int(res))
        if self.image_size == 512:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif self.image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif self.image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif self.image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            channel_mult = (1, 2, 4, 8)

        # self.model = UNetModel(self.image_size, self.in_channels, 128, self.out_channels, 2, tuple(attention_ds),
        #                        channel_mult=channel_mult, num_classes=None, dropout=0.25).to(self.device)

        self.model = UNetModel(self.image_size, self.in_channels, 64, self.out_channels, 2, tuple(attention_ds),
                               channel_mult=(1, 2, 2, 4, 8), num_classes=None, dropout=0.2).to(self.device)

        attention_resolutions = "32,16,8"
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(self.image_size // int(res))

        self.use_classifier = True
        # self.classifier = EncoderUNetModel(self.image_size, self.in_channels, 32, 2, 4, tuple(attention_ds),
        #                                    channel_mult=channel_mult, num_head_channels=32, use_scale_shift_norm=True,
        #                                    resblock_updown=True, pool='attention',).to(self.device)
        self.noise_level = 100

        self.diffusion = Diffusion(noise_steps=1000, img_size=self.image_size, device=self.device)

        # self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        self.optimizer = optim.AdamW(self.model.parameters(), 1e-4, weight_decay=25 * 1e-5)
        # self.classifier_optimizer = optim.AdamW(self.classifier.parameters(), lr=args.lr)

        self.start_time = time.strftime('%y_%m_%d_%H_%M_%S')
        # self.logger['info'] += 'from 4 channels to 1 using sum/'

    def prepare_train(self):
        self.mse = nn.MSELoss()
        # self.mse=ComboLoss(weights={'bce': 3, 'dice': 1, 'focal': 4, 'jaccard': 0, 'lovasz': 0, 'lovasz_sigmoid': 0})
        # mse = DistanceLoss(p=5)
        # cl_loss = nn.CrossEntropyLoss()
        self.cl_loss = nn.NLLLoss()

    def train_step(self, batch_idx, data, labels, path) -> torch.Tensor:
        t = self.diffusion.sample_timesteps(data.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(data, t)

        # predicted_noise = self.model(x_t, t, labels)
        predicted_noise = self.model(x_t, t)
        loss = self.mse(noise, predicted_noise)
        # pred = self.model(x_t, t)
        # loss = torch.mean(torch.mean(torch.square(data - pred), dim=[2, 3]) * t)
        # loss = mse(data, pred)
        # loss = 0.6 * mse(data, pred[:, 0, :].view(-1, 1, self.image_size, self.image_size))\
        #     + 0.4 * mse(noise, pred[:, 1, :].view(-1, 1, self.image_size, self.image_size))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # logits = self.classifier(x_t, t)
        # c_loss = cl_loss(F.log_softmax(logits, dim=1), labels)
        # # c_loss = cl_loss(logits, F.one_hot(labels.long(), num_classes=2).float())
        #
        # self.classifier_optimizer.zero_grad()
        # c_loss.backward()
        # self.classifier_optimizer.step()

        return loss

    def prepare_test(self):
        pass

    def test_step(self, index, x, labels, path) -> (torch.Tensor, torch.Tensor):
        # x_t = x
        # plot_img_to_file(x[0, 0, :, :], f'images/diffusion/orig.jpg')
        t = torch.ones(x.shape[0]).long().to(self.device)
        x_t, _ = self.diffusion.noise_images(x, t * self.noise_level)

        # pred = self.model(x_t, t)
        # x_t = pred[:, 0, :].view(-1, 1, self.image_size, self.image_size)

        # labels = torch.zeros(data.shape[0]).to(self.device).long()
        # labels = torch.randint(low=0, high=2, size=(data.shape[0],)).to(self.device).long()

        for j in reversed(range(1, self.noise_level+1)):
        # for j in [1, 250, 500, 750, 900, 750, 500, 250, 1]:
        #     plot_img_to_file(x_t[0, 0, :, :], f'images/diffusion/{j}.jpg')
            t = (torch.ones(x_t.shape[0]) * j).long().to(self.device)
            # x_t, _ = self.diffusion.noise_images(x_t, t)
            x_t = self.diffusion.reverse_noising(self.model, x_t, t)
            # x_t = self.diffusion.remove_noise(self.model, x_t, t)
            # x_t, noise = self.diffusion.noise_images(x_t, t)
            # pred = self.model(x_t, t)
            # x_t = pred[:, 0, :].view(-1, 1, self.image_size, self.image_size)
        # plot_img_to_file(x_t[0, 0, :, :], f'images/diffusion/final.jpg')
        # t = (torch.ones(data.shape[0]) * 10).long().to(self.device)
        # x_t, noise = self.diffusion.noise_images(x, t)
        # x = self.model(x_t, t)
        # x = pred[:, 0, :].view(-1, 1, self.image_size, self.image_size)

        # for i in [1000, 500, 100, 1]:
        #     t = (torch.ones(data.shape[0]) * i).long().to(self.device)
        #     logits = self.classifier(x, t)
        #     labels = F.softmax(logits, dim=1).argmax(-1)
        #     x = self.diffusion.reconstruct(self.model, x, t, labels)

        # for i in range(1, self.noise_level):
        #     t = (torch.ones(data.shape[0]) * i).long().to(self.device)
        #     # logits = self.classifier(x, t)
        #     # labels = F.softmax(logits, dim=1).argmax(-1)
        #     # x = self.diffusion.ddim_reverse_sample(self.model, x, t, labels)
        #     x = self.diffusion.ddim_reverse_sample(self.model, x, t)
        #     # pred = self.model(x_t, t)
        #     # x = pred[:, 0, :].view(-1, 1, self.image_size, self.image_size)
        #
        # for i in reversed(range(1, self.noise_level+1)):
        #     t = (torch.ones(data.shape[0]) * i).long().to(self.device)
        #     # _, scaled_grad = cond_fn(self.classifier, x, t, self.classifier_scale, labels)
        #     # logits = self.classifier(x, t)
        #     # labels = F.softmax(logits, dim=1).argmax(-1)
        #     # x = self.diffusion.ddim_sample(self.model, x, t, labels, scaled_grad)
        #     x = self.diffusion.ddpm_sample(self.model, x, t)

        # loss = loss_function(data, x[:, 0, :].view(-1, 1, self.image_size, self.image_size)).item()
        # + 0.4 * (loss_function(noise, x[:, 1, :].view(-1,1,self.image_size, self.image_size))).item()
        # loss = loss_function(x, x_t).item()
        loss = torch.mean(torch.square(x - x_t), dim=(1, 2, 3))
        # loss = torch.amax(torch.square(x - x_t), dim=(1, 2, 3))

        return loss, x_t

    # def load_model(self, model_name):
    #     state_dict = torch.load(model_name)
    #     if self.use_classifier:
    #         self.model.load_state_dict(state_dict['model'])
    #         self.classifier.load_state_dict(state_dict['classifier'])
    #     else:
    #         self.model.load_state_dict(state_dict)


def cond_fn(classifier, x, t, classifier_scale,  y=None):
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        a=torch.autograd.grad(selected.sum(), x_in)[0]
        return a, a * classifier_scale
