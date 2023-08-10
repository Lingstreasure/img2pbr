import hashlib
import os

import requests
from tqdm import tqdm

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
