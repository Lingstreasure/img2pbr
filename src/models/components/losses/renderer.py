from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.models.components.utils import color_input_check, grayscale_input_check


def AdotB(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    """Calculate vector a dot vector b."""
    return (a * b).sum(dim=dim, keepdim=True).clamp(min=0).expand(-1, 3, -1, -1)


def normalizeMap(map: torch.Tensor) -> torch.Tensor:
    """Normalize the input map from [-1, 1] to [0, 1]."""
    return map * 0.5 + 0.5


def norm(vec: torch.Tensor, dim: int = 0, eps: float = 1.0e-12) -> torch.Tensor:
    """Normalize the input vector or matrix along dim.

    :param vec: The input tensor of vector or matrix.
    :param dim: Along which dimension to do norm operation. Default to `0`.
    :param eps: Prevent to divide by zero. Default to `1.e-12`.
    """
    return vec / (torch.linalg.norm(vec, dim=dim, keepdim=True)).clamp_min(eps)


def getDir(pos: torch.Tensor, tex_pos: torch.Tensor) -> torch.Tensor:
    """Calculate the direction vector.

    :param pos: Target position at world-axis.
    :param tex_pos: Texture plane position at world-axis.
    :return: The tensor of direction vector.
    """
    vec = pos - tex_pos
    return norm(vec, dim=0), (vec**2).sum(dim=0, keepdim=True)


class DifferentiableRenderer(nn.Module):
    """Differentiable physics-based renderer using SVBRDF maps.

    Static members:
        CHANNELS (Dict[str, Tuple[bool, float]]): Supported types of output SVBRDF maps in a
            procedural material graph.
    """

    # Description of supported output channels and their format channel: (is color, default value)
    # where 'is color' is a bool - True = color; False = grayscale. default value is either 1.0 or 0.0
    CHANNELS: Dict[str, Tuple[bool, float]] = {
        "basecolor": (True, 0.0),
        "normal": (True, 1.0),
        "roughness": (False, 1.0),
        "metallic": (False, 0.0),
    }

    def __init__(
        self,
        size: float = 30.0,
        camera: List[float] = [0.0, 0.0, 25.0],
        light_color: List[float] = [3300.0, 3300.0, 3300.0],
        f0: float = 0.04,
        normal_format: str = "dx",
    ) -> None:
        """Initialize the differentiable renderer.

        :param size: Real-world size of the texture. Default to 30.0.
        :param camera: Position of the camera relative to the texture center.
            The texture always resides on the X-Y plane in center alignment. Default to `[0.0, 0.0, 25.0]`.
        :param light_color: Light intensity in RGB. Default to `[3300.0, 3300.0, 3300.0]`.
        :param f0: Normalized ambient light intensity. Default to `0.04`.
        :param normal_format: Controls how the renderer interprets the format of normal maps (DirectX 'dx' or
              OpenGL 'gl'). Default to `'dx'`.
        """
        super().__init__()

        # create necessary tensor data
        self.register_buffer("size", torch.as_tensor(size))
        self.register_buffer("light_color", torch.as_tensor(light_color).view(3, 1, 1))

        # Camera and base reflectivity are always fixed
        self.register_buffer("camera", torch.as_tensor(camera).view(3, 1, 1))
        self.register_buffer("f0", torch.as_tensor(f0))

        self.normal_format = normal_format

        # Multi lights: only use one light during rendering
        x_del = size / 4
        z = camera[-1]
        self.register_buffer(
            "light_poses",
            torch.stack(
                [
                    self.camera,
                    torch.tensor([-x_del, -x_del, z]).view(3, 1, 1),
                    torch.tensor([-x_del, x_del, z]).view(3, 1, 1),
                    torch.tensor([x_del, -x_del, z]).view(3, 1, 1),
                    torch.tensor([x_del, x_del, z]).view(3, 1, 1),
                ]
            ),
        )  # 5 positions around center point

    def DistributionGGX(
        self, n_dot_h: torch.Tensor, alpha: torch.Tensor, eps: float = 1.0e-12
    ) -> torch.Tensor:
        """A GGX normal distribution function term.

        :param cos: The vector of normal * half vector.
        :param alpha: Usually alpha = roughness ** 2.
        """
        n_dot_h2 = n_dot_h**2
        a2 = alpha**2
        denom = (n_dot_h2 * (a2 - 1.0) + 1.0) ** 2 * torch.pi
        return a2 / (denom + eps)

    def Fresnel(self, v_dot_h: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """A Fresnel-Schlick approximation term."""
        return f0 + (1.0 - f0) * torch.clamp((1 - v_dot_h), 0.0, 1.0) ** 5

    def Fresnel_S(self, v_dot_h: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """A Fresnel-Schlick approximation term used in UE4."""
        sphg = torch.pow(2.0, ((-5.55473 * v_dot_h) - 6.98316) * v_dot_h)
        return f0 + (1.0 - f0) * sphg

    def GSmith(
        self, n_dot_v: torch.Tensor, n_dot_l: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """A Geometry function term."""

        def _G1(cos, k):
            return cos / (cos * (1.0 - k) + k)

        k = (alpha * 0.5).clamp(min=1e-12)
        return _G1(n_dot_v, k) * _G1(n_dot_l, k)

    def forward(
        self,
        *tensors: torch.Tensor,
        normalized: bool = True,
        use_diffuse: bool = True,
        use_metallic: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Generate a rendered image from SVBRDF maps of an input texture.

        :param tensors: Sequence of input SVBRDF maps. Each map is interpreted per the order defined in
              `DifferentiableRenderer.CHANNELS` and with shape of [B, C, H, W].
        :param normalized: Input SVBRDF maps are normalized to [0, 1], otherwise [-1, 1]. Default to `True`.
        :param use_diffuse: Whether to use diffuse term. Default to `True`.
        :param use_metallic: Whether to use metallic map. Default to `True`.
        :param eps: A small epsilon that thresholds denominators to prevent division by zero. Default to `1e-12`.

        :return: Tensor of rendered image using input SVBRDF maps.
        """
        # Check input validity
        for i, (label, (is_color, _)) in enumerate(self.CHANNELS.items()):
            if i >= len(tensors):
                break
            check_func = color_input_check if is_color else grayscale_input_check
            check_func(tensors[i], label)

        # Normalize the SVBRDF maps
        albedo, normal, roughness = tensors[:3]

        if not normalized:
            albedo = normalizeMap(albedo)
            roughness = normalizeMap(roughness)
        else:  # normal[0, 1] -> [-1, 1]
            normal = (normal - 0.5) * 2.0

        # Map the basecolor to gamma space
        albedo = albedo**2.2

        # Process DirectX normal format by default
        if self.normal_format == "dx":
            normal = normal * torch.tensor([1, -1, 1], device=normal.device).view(1, 3, 1, 1)

        # Account for metallicity - increase base reflectivity and decrease albedo
        f0 = self.f0
        if use_metallic:
            metallic = tensors[3] if normalized else normalizeMap(tensors[3])
            f0 = torch.lerp(self.f0, albedo, metallic)
            albedo = albedo * (1.0 - metallic)

        # Calculate 3D pixel center positions (image lies on the x-y plane)
        # img_size: int = albedo.shape[2]
        H, W = albedo.shape[-2:]
        x_coords = torch.linspace(0.5 / W - 0.5, 0.5 - 0.5 / W, W, device=albedo.device)
        x_coords = x_coords * self.size
        y_coords = torch.linspace(0.5 / H - 0.5, 0.5 - 0.5 / H, H, device=albedo.device)
        y_coords = y_coords * self.size

        x, y = torch.meshgrid(x_coords, y_coords, indexing="xy")
        pos = torch.stack((x, -y, torch.zeros_like(x)))  # x-axis: >, y-axis: ^

        # Calculate view directions (half vectors from camera to each pixel center)
        v, _ = getDir(self.camera, pos)
        light_pos = self.light_poses[torch.randint(0, 5, (1,))]  # random select one light
        l, distance2 = getDir(light_pos, pos)
        h = norm(l + v, dim=0)
        normal = norm(normal, dim=1)

        n_dot_v = AdotB(normal, v, dim=1)
        n_dot_l = AdotB(normal, l, dim=1)
        n_dot_h = AdotB(normal, h, dim=1)
        v_dot_h = AdotB(v, h, dim=0)

        D = self.DistributionGGX(n_dot_h, roughness**2)
        F = self.Fresnel_S(v_dot_h, f0)
        G = self.GSmith(n_dot_v, n_dot_l, roughness**2)

        geometry_times_light = n_dot_h * self.light_color / distance2.clamp_min(eps)

        # Get the diffuse term with lambert brdf
        diffuse = albedo / torch.pi

        # Get the specular term with cook-torrance brdf
        specular = D * F * G / (4 * n_dot_v * n_dot_l).clamp_min(eps)

        rendering = torch.clamp(specular * geometry_times_light, eps, 1.0)  # eps: prevent nan

        # Whether use diffuse term
        if use_diffuse:
            rendering = torch.clamp(
                rendering + diffuse * geometry_times_light, eps, 1.0
            )  # eps: prevent nan

        # De-gamma
        rendering = rendering ** (1 / 2.2)  # prevent rendering to be zero
        return rendering
