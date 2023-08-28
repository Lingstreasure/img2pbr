import hashlib
import os
from typing import Sequence

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

Class_Dict = {0: "", 1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: ""}

URL_MAP = {"vgg": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

CKPT_MAP = {"vgg": "vgg.pth", "alex": "alex.pth"}

MD5_MAP = {"vgg": "d507d7349b931f0638a25a48a722f98a"}


# def download(url, local_path, chunk_size=1024):
#     "Download the file from the arg `local_path`."
#     os.makedirs(os.path.split(local_path)[0], exist_ok=True)
#     with requests.get(url, stream=True) as r:
#         total_size = int(r.headers.get("content-length", 0))
#         with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
#             with open(local_path, "wb") as f:
#                 for data in r.iter_content(chunk_size=chunk_size):
#                     if data:
#                         f.write(data)
#                         pbar.update(chunk_size)


# def md5_hash(path):
#     """Mapping the md5 code."""
#     with open(path, "rb") as f:
#         content = f.read()
#     return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    """Get the ckpt path with arg `name` and it's `root` directory."""
    # assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    # if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
    #     print(f"Downloading {name} model from {URL_MAP[name]} to {path}")
    #     download(URL_MAP[name], path)
    #     md5 = md5_hash(path)
    #     assert md5 == MD5_MAP[name], md5
    return path


def log_txt_as_img(wid_hei: Sequence[int], texts: Sequence[int], size: int = 30) -> torch.Tensor:
    """Turn a list of texts to a image-like tensor.

    :param wid_hei: The width and height of target image tensor.
    :param texts: A sequence of texts to plot.
    """

    batch_size = len(texts)
    txts = list()
    num_cnt = int(40 * (wid_hei[0] / 256))
    for b_idx in range(batch_size):
        txt = Image.new("RGB", wid_hei, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        lines = "\n".join(
            texts[b_idx][start : start + num_cnt] for start in range(0, len(texts[b_idx]), num_cnt)
        )

        try:
            draw.text((wid_hei[0] // 4, wid_hei[0] // 4), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Can't encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0  # 0-255 -> -1.-1.
        txts.append(txt)

    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def color_input_check(img: torch.Tensor, var_name: str):
    """Verify a tensor input represents a color image.

    :param img (Tensor): Source tensor input.
    :param var_name (str): Argument name of the tensor.
    :return: raise ValueError - The input tensor does not represent a color image.
    """
    if not isinstance(img, torch.Tensor) or img.ndim != 4 or img.shape[1] not in (3, 4):
        raise ValueError(
            f"Node function input '{var_name}' must be a color image but have "
            f"{img.shape[1]} channels"
        )


def grayscale_input_check(img: torch.Tensor, var_name: str):
    """Verify a tensor input represents a grayscale image.

    :param img (Tensor): Source tensor input.
    :param var_name (str): Argument name of the tensor.
    :return: raise ValueError - The input tensor does not represent a grayscale image.
    """
    if not isinstance(img, torch.Tensor) or img.ndim != 4 or img.shape[1] != 1:
        raise ValueError(
            f"Node function input '{var_name}' must be a grayscale image but "
            f"have {img.shape[1]} channels"
        )
