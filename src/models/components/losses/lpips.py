"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""
from collections import namedtuple
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
from torchvision import models

from src.models.components.utils import get_ckpt_path


class Vgg16(torch.nn.Module):
    """Vgg16 implementation."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> NamedTuple:
        """Process input."""

        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class Alexnet(torch.nn.Module):
    """Alexnet implementation."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        alexnet_pretrained_features = models.alexnet(pretrained=pretrained).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> NamedTuple:
        """Process input."""

        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple(
            "alexnet_outputs", ["relu1", "relu2", "relu3", "relu4", "relu5"]
        )
        return alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


def normalize_tensor(x, eps=1e-10):
    """Normalize the input tensor from [-1, 1] to [0, 1]."""

    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    """Average the tensor in spatial dimension."""

    return x.mean([2, 3], keepdim=keepdim)


class ScalingLayer(nn.Module):
    """A layer using for scaling the input tensor."""

    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Process the input."""

        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class LPIPS(nn.Module):
    """The Learned Perceptual Image Patch Similarity (`LPIPS`) calculates the perceptual similarity
    between two images.

    LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network.
    This measure has been shown to match human perception well. A low LPIPS score means that image patches are
    perceptual similar.

    Both input image patches are expected to have shape ``(N, 3, H, W)``. The minimum size of `H, W` depends on the
    chosen backbone (see `net` arg).
    """

    def __init__(self, net: Literal["alex", "vgg"] = "alex", use_dropout: bool = True) -> None:
        super().__init__()

        self.scaling_layer = ScalingLayer()

        net_type = net
        if net_type in ["vgg", "vgg16"]:
            net_type = Vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif net_type == "alex":
            net_type = Alexnet  # type: ignore[assignment]
            self.chns = [64, 192, 384, 256, 256]

        self.net = net_type(pretrained=True, requires_grad=False)

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained(name=net)
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg"):
        """Load pretrained model path according to `name` arg."""
        ckpt = get_ckpt_path(name, "src/models/lpips_models")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print(f"loaded pretrained LPIPS loss from {ckpt}")

    def forward(self, input, target):
        """Process the input tensor and target tensor to cal the similarity."""
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for idx in range(1, len(self.chns)):
            val = val + res[idx]
        return val
