from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from xformers.components.attention import LinformerAttention

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class ConvBlock(nn.Module):
    """A ConvBlock is as same architecture of ResdualBlock."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 0,
        use_batch_norm: bool = False,
        num_groups: int = 16,
        *args, 
        **kwargs
    ) -> None:
        """Initialize a `ConvBlock`
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels. Default to `0`.
        :param num_group: The number of normalization group. Default to `16`.
        """
        super().__init__(*args, **kwargs)
        
        self.in_channels = in_channels
        out_channels = in_channels if out_channels == 0 else out_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        if use_batch_norm:
            self.norm1 = nn.BatchNorm2d(out_channels)  
        else:
            self.norm1 = Normalize(out_channels, num_groups=num_groups)
        
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        if use_batch_norm:
            self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            self.norm2 = Normalize(out_channels, num_groups=num_groups)
            
        
        self.skip = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1)
        if use_batch_norm:
            self.skip_norm = nn.BatchNorm2d(out_channels)
        else:
            self.skip_norm = Normalize(out_channels, num_groups=num_groups)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.conv1(h)
        h = self.norm1(h)
        h = F.silu(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        
        x = self.skip(x)
        # x = self.skip_norm(x)
        return h + x
    
    
class MLP(nn.Module):
    """2 Convolutional layers based Multi Layer Perceptron."""
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = False,
        num_groups: int = 16, 
        dropout: float = 0.2, 
        *args, 
        **kwargs
    ) -> None:
        """Initialize a `MLP`
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels. Default to `3`.
        :param use_batch_norm: Whether to use Batch Normalization. Default to `False`.
        :param num_group: The number of normalization group. Default to `16`.
        :param dropout: The probability of Dropout layer.
        """
        super().__init__(*args, **kwargs)
        
        middle_channels = 3 * in_channels
        self.conv_in = nn.Conv2d(in_channels,
                                 middle_channels,
                                 kernel_size=1,
                                 stride=1)
        if use_batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = Normalize(middle_channels, num_groups=num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv2d(middle_channels,
                                  out_channels,
                                  kernel_size=1,
                                  stride=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = F.silu(x)
        return self.conv_out(x)
    

# class LinearSelfAttention(nn.Module):
#     """A Linear Self-Attention layer for image-like data.
    
#     This implementation is based on xformer LinformerAttention.
#     First, project the input (aka embedding) and reshape to b, t, d. 
#     Then apply standard transformer action.
#     Finally, reshape to image。
#     """
    
#     def __init__(
#         self, 
#         in_channels,
#         num_heads: int = 4,
#         dim_head: int = 32,
#         depth: int = 1,
#         dropout: float = 0.,
#         *args, 
#         **kwargs
#     ) -> None:
#         """Initialize a `LinformerBlock`.
#         :param in_channels: The number of input channels.
#         :param n_heads: The number of multi-heads.
#         :param dim_head: The dim of q, k, v.
#         """
#         super().__init__(*args, **kwargs)
#         self.num_heads = num_heads
#         inner_dim = num_heads * dim_head
#         self.attention = LinformerAttention(dropout=dropout, seq_len= , k=)
#         self.to_qkv = nn.Conv2d(in_channels, inner_dim * 3, 1, bias = False)
#         self.to_out = nn.Conv2d(inner_dim, in_channels, 1)
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x)  # b c h w -> b (3 * h * d) h w
#         q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        

class LinearAttention(nn.Module):
    """A self attention module with linear complexity.
    
    This implementation is from openaimodel.py in SD.
    """
    
    def __init__(
        self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Upsample(nn.Module):
    """An upsampling layer with an optional convolution."""

    def __init__(
        self, 
        in_channels: int, 
        use_conv: bool, 
        dims: int = 2, 
        out_channels: Optional[int] = None, 
        mode: str = "nearest",
        padding=1
    ) -> None:
        """Initialize a Upsample module.
        
        :param in_channels: Channels in the inputs and outputs.
        :param use_conv: A bool determining if a convolution is applied.
        :param dims: Determines if the signal is 1D, 2D, or 3D. If 3D, then 
                     upsampling occurs in the inner-two dimensions. Default to `2`.
        :param out_channels: Channels in the outputs. Default to `None`.
        :param mode: Mode in interpolation of upsampling. Default to `nearest`.
        :param padding: padding of convolution layer. Default to `1`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.mode = mode
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.in_channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode=self.mode
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)
        if self.use_conv:
            x = self.conv(x)
        return x


class Upsample2D(nn.Module):
    """An upsampling layer with an 2D convolution."""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: Optional[int] = None, 
        mode: str = "nearest",
        padding: int = 1
    ) -> None:
        """Initialize a Upsample2D module.
        
        :param in_channels: Channels in the inputs and outputs.
        :param out_channels: Channels in the outputs. Default to `None`.
        :param mode: Mode in interpolation of upsampling. Default to `nearest`.
        :param padding: padding of convolution layer. Default to `1`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.mode = mode
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode=self.mode)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """A downsampling architecture of neural network."""

    def __init__(
        self, 
        *args, 
        **kwargs
    ) -> None:
        """
        """
        super().__init__(*args, **kwargs)
    
    
class Decoder(nn.Module):
    """A upsampling architecture of neural network."""
    
    def __init__(
        self, 
        *args, 
        **kwargs
    ) -> None:
        """
        """
        super().__init__(*args, **kwargs)
        
        
