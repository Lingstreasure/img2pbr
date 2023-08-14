# reference: https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L26
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.basic_module import Normalize


class BasicConv(nn.Module):
    """Basic convolutional block.

    The block consists of convolutional layer, normalization layer and activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 0,
        groups: int = 1,
        bias: bool = True,
        activate: bool = True,
        num_groups: int = 16,
    ) -> None:
        """Initialize a BasicConv Module.

        :param 1~8: Same with the pytorch nn.Conv2d module.
        :param activate: Whether to use a activation layer. Default to `True`.
        :param num_groups: The number of normalization group. Default to `16`.
        """
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = Normalize(out_channels, num_groups=num_groups)
        self.activate = nn.SiLU() if activate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implement the forward() function."""
        x = self.conv(x)
        x = self.norm(x)
        if self.activate:
            x = self.activate(x)
        return x


class ChannelGate(nn.Module):
    """The Channel Attention Module.

    A module for channel-based attention in convolutional neural networks. Introduced by Woo et al.
    in CBAM: Convolutional Block Attention Module.
    https://paperswithcode.com/paper/cbam-convolutional-block-attention-module
    """

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]) -> None:
        """ "Initialize a ChannelGate module.

        :param gate_channels: The channel of input feature.
        :param reduction_ratio: The reduction ratio between gate_channels and hidden units number
            of MLP.
        """
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.SiLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate channel attention of input tensor."""
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor: torch.Tensor) -> torch.Tensor:
    """A pool method for log(sum(exp(x)))."""
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    """The Channel Pool module.

    A module extract the max and mean value on channel dimension.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the information of input tensor."""
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    """Spatial Attention Module.

    A module for spatial attention in convolutional neural networks.Introduced by Woo et al.
    in CBAM: Convolutional Block Attention Module.
    https://paperswithcode.com/paper/cbam-convolutional-block-attention-module
    """

    def __init__(self) -> None:
        """Initialize a SpatialGate module."""
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, activate=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate spatial attention of input tensor."""
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    an attention module for convolutional neural networks. Given an intermediate feature map, the
    module sequentially infers attention maps along two separate dimensions, channel and spatial,
    then the attention maps are multiplied to the input feature map for adaptive feature
    refinement.

    source:
    https://paperswithcode.com/method/cbam
    """

    def __init__(
        self,
        gate_channels,
        reduction_ratio: int = 16,
        pool_types: Sequence[str] = ["avg", "max"],
        no_spatial: bool = False,
    ) -> None:
        """Initialize a CBAM module.

        :param gate_channels: The channel of input feature.
        :param reduction_ratio: The reduction ratio between gate_channels and hidden units number of MLP.
               Default to `16`.
        :param pool_types: The sequence of pool types used for Channel Attention Module. Default to
               `['avg', 'max']`.
        :param no_spatial: Do not use the Spatial Attention Module. Default to `False`.
        """
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the attention of input tensor x and multiply it."""
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
