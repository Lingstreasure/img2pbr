from typing import List, Tuple, Union, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.basic_module import ConvBlock, MLP, LinearAttention
from src.models.components.mobile_vit import MobileVit

class UNetModel(nn.Module):
    """A modified UNet model with linear attention and MobileViT. """
    
    def __init__(
        self, 
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 32, 
        num_conv_blocks: Union[int, Sequence[int]] = 1,
        channel_mult: Sequence[int] = (1, 1, 1, 1),
        use_self_attention: bool = True,
        use_ViT_bottleneck: bool = True,
        use_MLP_out: bool = True,
        use_fp16: bool = False,
        *args, 
        **kwargs
    ) -> None:
        """Initialize a `UNetModel`.
        
        :param in_channels: Channels in the input Tensor. Default to `3`.
        :param out_channels: Channels in the output Tensor. Default to `3`.
        :param model_channels: Base channel count for the model
        :param num_conv_blocks: Number of conv blocks(residual blocks) per downsample.
        :param channel_mult: Channel multiplier for each level of the UNet. Default to `(1, 1, 1, 1)`.
        :param use_self_attention: Whether to add linear self attention after conv blocks. Default to `True`.
        :param use_ViT_bottleneck: Whether to use MobileViT at bottleneck. Default to `True`.
        :param use_MLP_out: Whether to use MLP as the output layer. Default to `True`.
        :param use_fp16: Whether to use fp16 with mixed precision (not implemented). Default to `False`.
        """
        super().__init__(*args, **kwargs)
        
        if isinstance(num_conv_blocks, int):
            self.num_conv_blocks = len(channel_mult) * [num_conv_blocks]
        else:
            if len(num_conv_blocks) != len(channel_mult):
                raise ValueError("provide num_conv_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_conv_blocks = num_conv_blocks
        
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        self.input_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        fea_channels = [32 * pow(2, level) for level in range(len(channel_mult))]
 
        # downsampling
        layers = []
        for ch in fea_channels:
            layers.append(ConvBlock(in_channels, ch))
            if use_self_attention:
                layers.append(LinearAttention(dim=ch))
            self.input_blocks.append(nn.Sequential(*layers))
            layers.clear()
            in_channels = ch
 
        for ch in reversed(fea_channels):
            layers.append(nn.ConvTranspose2d(in_channels=ch, out_channels=ch // 2, kernel_size=2, stride=2))
            layers.append(ConvBlock(in_channels=ch, out_channels=ch // 2))
            self.output_blocks.append(nn.ModuleList(layers))
            self.skip_blocks.append(nn.Conv2d(in_channels=ch, out_channels=ch // 2, kernel_size=1)) 
            layers.clear()
                
 
        if use_ViT_bottleneck:
            self.middle_block = MobileVit()
        else:
            self.middle_block = ConvBlock(in_channels, in_channels)
        
        if use_MLP_out:
            self.out = MLP(in_channels=fea_channels[0] // 2, out_channels=out_channels)
        else:
            self.out = ConvBlock(in_channels=fea_channels[0] // 2, out_channels=out_channels)
        
    def forward(self, x) -> torch.Tensor:
        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)  # encode
            hs.append(h)
            h = self.pool(h)  # downsampling
            
        h = self.middle_block(h)
        for skip_module, modules in zip(self.skip_blocks, self.output_blocks):
            skip_fea = skip_module(hs.pop())
            h = modules[0](h)  # upsampling
            h = torch.cat([h, skip_fea], dim=1)
            h = modules[1](h)  # decode
                
        h = self.out(h)
        return h.type(x.dtype)
    
    
if __name__ == "__main__":
    model = UNetModel(in_channels=3,
                      num_conv_blocks=1,
                      channel_mult=(1, 1, 1, 1),
                      use_self_attention=True,
                      use_ViT_bottleneck=False,
                      use_MLP_out=True,
                      use_fp16=False)
    print(model)
    cnt = sum(p.numel() for p in model.parameters())
    print(cnt, '\t', "{:.2f}".format(cnt / 1000000), 'M')
    imgs = torch.randn((4, 3, 512, 512))
    out = model(imgs)
    print(out.shape)

    