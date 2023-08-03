import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from xformers.components.attention import LinformerAttention

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ConvBlock(nn.Module):
    """A ConvBlock is as same architecture of ResdualBlock."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 0,
        *args, 
        **kwargs
    ) -> None:
        """Initialize a `ConvBlock`
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels. Default to `0`.
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
        self.norm1 = Normalize(out_channels, num_groups=16)
        
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm2 = Normalize(out_channels, num_groups=16)
        
        self.skip = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1)
        self.skip_norm = Normalize(out_channels, num_groups=16)
        
    def forward(self, x) -> torch.Tensor:
        h = x
        h = self.conv1(h)
        h = self.norm1(h)
        h = F.silu(h) 
        
        h = self.conv2(h)
        h = self.norm2(h)
        
        x = self.skip(x)
        x = self.skip_norm(x)
        return h + x
    
    
class MLP(nn.Module):
    """2 Convolutional layers based Multi Layer Perceptron."""
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2, 
        *args, 
        **kwargs
    ) -> None:
        """Initialize a `MLP`
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels. Default to `3`.
        :param dropout: The probability of Dropout layer
        """
        super().__init__(*args, **kwargs)
        
        middle_channels = 3 * in_channels
        self.conv_in = nn.Conv2d(in_channels,
                                 middle_channels,
                                 kernel_size=1,
                                 stride=1)
        self.norm = Normalize(middle_channels, num_groups=16)
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv2d(middle_channels,
                                  out_channels,
                                  kernel_size=1,
                                  stride=1)
        
    def forward(self, x) -> torch.Tensor:
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
        
#     def forward(self, x) -> torch.Tensor:
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x)  # b c h w -> b (3 * h * d) h w
#         q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


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
        
        
