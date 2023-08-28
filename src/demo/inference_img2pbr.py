import argparse
import math
from typing import List, Literal, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def closest_power_of_two(num: int) -> int:
    """Find the cloest 2^x of input number."""
    log_num = math.log(num, 2)
    lower_power = 2 ** math.floor(log_num)
    upper_power = 2 ** math.ceil(log_num)
    if abs(lower_power - num) < abs(upper_power - num):
        return lower_power
    else:
        return upper_power


def load_model_from_config(
    config: DictConfig,
    ckpt: str,
    model_type: Literal["generator", "model"] = "generator",
    verbose: bool = False,
) -> torch.nn.Module:
    """Load the model from the config file and pytorch-lightning ckpt file.

    :param config: The config file.
    :param ckpt: The path of pytorch-lightning ckeckpoint file.
    :param model_type (optional): The model type in LightningModule (GAN or basemodel).
        Default to `generator`.
    :param verbose: Whether to show messages in terminal. Default to `False`.
    :return: Loaded torch model.
    """
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)
    sd = pl_sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("G."):
            new_sd[k.replace("G.", "")] = v
    del sd
    model = hydra.utils.instantiate(config.get(model_type))
    m, u = model.load_state_dict(new_sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


parser = argparse.ArgumentParser()
opt = parser.parse_args()

# opt.C = 4
# opt.H = 512
# opt.W = 512
# opt.f = 8

opt.device_num = 1
opt.config = "configs/model/pbr_gan.yaml"
opt.pbr_ckpt = (
    "logs/pbr_reconstruction/train_gan/runs/2023-08-16_17-20-42/checkpoints/epoch_053.ckpt"
)

opt.seed = 42
pl.seed_everything(opt.seed)
# TODO seed -1

device = (
    torch.device(f"cuda:{opt.device_num}") if torch.cuda.is_available() else torch.device("cpu")
)

config = OmegaConf.load(f"{opt.config}")
pbr_model = load_model_from_config(config, opt.pbr_ckpt, "generator")
pbr_model = pbr_model.to(device)

img_transforms = transforms.Compose(
    [
        transforms.ToTensor(),  # row_data->(0, 1.0), h w c -> c h w
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # (0, 1.0)->(-1.0, 1.0)
    ]
)


def crop(img: np.ndarray, crop_sz: int, step: int) -> Tuple[List[np.ndarray], int]:
    """Crop the input image to patches."""
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError(f"Wrong image shape - {n_channels}")
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    patch_list = []
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x : x + crop_sz, y : y + crop_sz]
            else:
                crop_img = img[x : x + crop_sz, y : y + crop_sz, :]
            patch_list.append(crop_img)

    h = x + crop_sz
    w = y + crop_sz
    return patch_list, num_h, num_w, h, w


def combine(
    sr_list: List[np.ndarray], num_h: int, num_w: int, h: int, w: int, patch_size: int, step: int
):
    """Combine the patches with specific patch number and step."""
    index = 0
    out_img = np.zeros((h, w, 3), "float32")
    for i in range(num_h):
        for j in range(num_w):
            out_img[
                i * step : i * step + patch_size, j * step : j * step + patch_size, :
            ] += sr_list[index]
            index += 1
    out_img = out_img.astype("float32")

    for j in range(1, num_w):
        out_img[:, j * step : j * step + (patch_size - step), :] /= 2

    for i in range(1, num_h):
        out_img[i * step : i * step + (patch_size - step), :, :] /= 2
    return out_img.astype(np.uint8)


def img2pbr(img: Image.Image) -> Tuple[Image.Image]:
    """Predict the input image to albedo/normal/roughness map."""
    assert isinstance(img, Image.Image)  # and RGB

    w, h = img.size
    new_w = closest_power_of_two(w)
    new_h = closest_power_of_two(h)
    img = img.resize((new_w, new_h))

    img = np.array(img)

    input = img_transforms(img)[None]
    input = input.to(device)

    with torch.no_grad():
        pbr_maps = pbr_model(input)
        pbr_maps = ((pbr_maps + 1.0) / 2.0).float().clamp_(0, 1)

        albedo_rec = pbr_maps[:, :3, ...]
        normal_rec = pbr_maps[:, 3:6, ...]
        rough_rec = pbr_maps[:, 6:7, ...].repeat(1, 3, 1, 1)
        # metal_rec = pbr_maps[:, 7:, ...].repeat(1, 3, 1, 1)

        albedo = 255.0 * albedo_rec[0].permute(1, 2, 0).cpu().numpy()
        albedo = Image.fromarray(albedo.astype(np.uint8))

        normal = 255.0 * normal_rec[0].permute(1, 2, 0).cpu().numpy()
        normal = Image.fromarray(normal.astype(np.uint8))

        rough = 255.0 * rough_rec[0].permute(1, 2, 0).cpu().numpy()
        rough = Image.fromarray(rough.astype(np.uint8))

    return albedo, normal, rough


def _inference(img: Image.Image) -> Tuple[Image.Image]:
    """The inference process of img2pbr."""
    input = img_transforms(img)[None]
    input = input.to(device)

    with torch.no_grad():
        pbr_maps = pbr_model(input)
        pbr_maps = ((pbr_maps + 1.0) / 2.0).float().clamp_(0, 1)

        albedo_rec = pbr_maps[:, :3, ...]
        normal_rec = pbr_maps[:, 3:6, ...]
        rough_rec = pbr_maps[:, 6:7, ...].repeat(1, 3, 1, 1)
        # metal_rec = pbr_maps[:, 7:, ...].repeat(1, 3, 1, 1)

        albedo = 255.0 * albedo_rec[0].permute(1, 2, 0).cpu().numpy()
        albedo = Image.fromarray(albedo.astype(np.uint8))

        normal = 255.0 * normal_rec[0].permute(1, 2, 0).cpu().numpy()
        normal = Image.fromarray(normal.astype(np.uint8))

        rough = 255.0 * rough_rec[0].permute(1, 2, 0).cpu().numpy()
        rough = Image.fromarray(rough.astype(np.uint8))

    return albedo, normal, rough


def high_res_img2pbr(img: Image.Image) -> Tuple[Image.Image]:
    """Predict the high resolution input image to albedo/normal/roughness map."""
    assert isinstance(img, Image.Image)  # and RGB
    patch_sz = 512
    step = 512

    w, h = img.size
    # If input.size <= 512, resize it to 512.
    if w <= 512 or h <= 512:
        img = img.resize((512, 512))
        img = np.asarray(img)
        return _inference(img)
    else:  # execute the high resolution reconstruction code
        img = np.asarray(img)
        patches, num_h, num_w, h, w = crop(img, patch_sz, step)
        albedo_list = []
        normal_list = []
        rough_list = []
        for idx, patch in enumerate(patches):
            albedo, normal, rough = _inference(patch)
            albedo_list.append(np.asarray(albedo))
            normal_list.append(np.asarray(normal))
            rough_list.append(np.asarray(rough))

        albedo = combine(albedo_list, num_h, num_w, h, w, patch_sz, step)
        normal = combine(normal_list, num_h, num_w, h, w, patch_sz, step)
        rough = combine(rough_list, num_h, num_w, h, w, patch_sz, step)
        return albedo, normal, rough


if __name__ == "__main__":
    imgf = "logs/pbr_reconstruction/predict/runs/2023-08-17_10-15-56/outputs/24/input.jpg"
    img = Image.open(imgf).convert("RGB")  # PIL.Image，h w c

    albedo, normal, rough = img2pbr(img)

    albedo.save("src/demo/test_albedo.png")
    normal.save("src/demo/test_normal.png")
    rough.save("src/demo/test_rough.png")

    # import pdb;pdb.set_trace()
