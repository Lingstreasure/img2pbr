from typing import Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.basic_module import (
    MLP,
    ConvBlock,
    LinearAttention,
    Spectral_Normalization,
    Upsample2D,
)
from src.models.components.cbam import CBAM


class UNetDiscriminator(nn.Module):
    """A discriminator based on modified multi-decoder UNet model."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Sequence[int] = (3, 3, 1, 1),
        model_channels: int = 32,
        num_conv_blocks: Union[int, Sequence[int]] = 1,
        channel_mult: Sequence[int] = (1, 1, 1, 1),
        num_groups: int = 16,
        use_batch_norm: bool = False,
        use_self_attention: bool = False,
        use_ViT_bottleneck: bool = False,
        use_interpolation: bool = False,
        use_MLP_out: bool = False,
        use_spectral_norm: bool = True,
        use_fp16: bool = False,
    ) -> None:
        """Initialize a `UNetDiscriminator`.

        :param in_channels: Channels in the input Tensor. Default to `3`.
        :param out_channels: Channels in the multi output Tensors. Default to `(3, 3, 1)`.
        :param model_channels: Base channel count for the model.
        :param num_conv_blocks: Number of conv blocks(residual blocks) per downsample.
        :param channel_mult: Channel multiplier for each level of the UNet. Default to
                             `(1, 1, 1, 1)`.
        :param num_group: The number of normalization group. Default to `16`.
        :param use_batch_norm: Whether to use Batch Normalization. Default to `False`.
        :param use_self_attention: Whether to add linear self attention after conv blocks.
                                   Default to `False`.
        :param use_ViT_bottleneck: Whether to use Attention at bottleneck. Default to `False`.
        :param use_interpolation: Whether to use interpolation for upsampling in decoder.
                                  Default to `False`.
        :param use_MLP_out: Whether to use MLP as the output layer. Default to `False`.
        :param use_spectral_norm: Whether to use spectral normalization for nn.Linear and nn.Conv2d.
                                  Default to `True`.
        :param use_fp16: Whether to use fp16 with mixed precision (not implemented).
                         Default to `False`.
        """
        super().__init__()

        if isinstance(num_conv_blocks, int):
            self.num_conv_blocks = len(channel_mult) * [num_conv_blocks]
        else:
            if len(num_conv_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_conv_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_conv_blocks = num_conv_blocks

        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.out_channels = out_channels

        self.input_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()
        self.outs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        fea_channels = [model_channels * pow(2, level) for level in range(len(channel_mult))]

        # downsampling
        layers = []
        for ch in fea_channels:
            layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=ch,
                    use_batch_norm=use_batch_norm,
                    num_groups=num_groups,
                )
            )
            if use_self_attention:
                layers.append(LinearAttention(dim=ch))
            self.input_blocks.append(nn.Sequential(*layers))
            self.skip_blocks.append(nn.Conv2d(in_channels=ch, out_channels=ch // 2, kernel_size=1))
            layers.clear()
            in_channels = ch

        # bottleneck module
        if use_ViT_bottleneck:
            self.middle_block = CBAM(gate_channels=in_channels)
        else:
            self.middle_block = ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                use_batch_norm=use_batch_norm,
                num_groups=num_groups,
            )

        # the encoder out
        self.middle_out = nn.Linear(in_features=in_channels, out_features=1)

        # multi decoders
        for out_ch in out_channels:
            # out module of Model
            if use_MLP_out:
                self.outs.append(
                    MLP(
                        in_channels=fea_channels[0] // 2,
                        out_channels=out_ch,
                        use_batch_norm=use_batch_norm,
                        num_groups=num_groups,
                    )
                )
            else:
                self.outs.append(
                    nn.Conv2d(in_channels=fea_channels[0] // 2, out_channels=out_ch, kernel_size=1)
                )

            # upsampling
            self.output_blocks.append(nn.ModuleList())
            for ch in reversed(fea_channels):
                if use_interpolation:
                    layers.append(Upsample2D(in_channels=ch, out_channels=ch // 2, mode="nearest"))
                else:
                    layers.append(
                        nn.ConvTranspose2d(
                            in_channels=ch, out_channels=ch // 2, kernel_size=2, stride=2
                        )
                    )
                layers.append(
                    ConvBlock(
                        in_channels=ch,
                        out_channels=ch // 2,
                        use_batch_norm=use_batch_norm,
                        num_groups=num_groups,
                    )
                )
                self.output_blocks[-1].append(nn.ModuleList(layers))
                layers.clear()

        # use spectral normalization
        if use_spectral_norm:
            self.apply(Spectral_Normalization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor: x, output multi maps with decoders."""
        hs = []
        h = x.type(self.dtype)

        # encoder
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
            h = self.pool(h)  # downsampling

        # bottleneck
        h = self.middle_block(h)
        bottle_neck_out = self.middle_out(torch.sum(F.silu(h), dim=(2, 3)))

        # skip connection
        for level, module in enumerate(self.skip_blocks):
            hs[level] = module(hs[level])
        hs.reverse()

        # multi decoders
        outs = [h] * len(self.outs)
        for idx, decoder in enumerate(self.output_blocks):
            for level, modules in enumerate(decoder):
                outs[idx] = modules[1](torch.cat([modules[0](outs[idx]), hs[level]], dim=1))

            outs[idx] = self.outs[idx](outs[idx]).type(x.dtype)
            outs[idx] = torch.tanh(outs[idx])

        return bottle_neck_out, torch.concat(outs, dim=1)


if __name__ == "__main__":
    model = UNetDiscriminator(
        in_channels=3,
        out_channels=[3, 3],
        num_conv_blocks=1,
        channel_mult=(1, 1, 1, 1),
        use_self_attention=False,
        use_ViT_bottleneck=True,
        use_interpolation=False,
        use_MLP_out=False,
        use_fp16=False,
    )
    print(model)
    cnt = sum(p.numel() for p in model.parameters())
    print(cnt, "\t", f"{cnt / 1000000:.2f}", "M")
    imgs = torch.randn((4, 3, 512, 512))
    middle_out, outs = model(imgs)
    print(outs.shape)
