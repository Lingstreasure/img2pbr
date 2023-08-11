import os

import hydra
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from PIL import Image

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


def predict(cfg: DictConfig) -> torch.Tensor:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    log.info("Starting predicting!")
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    return predictions


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict_pbr.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for prediction.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    predictions = predict(cfg)

    # create output directory
    out_dir = os.path.join(cfg.paths.output_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # calculate the idx of different maps
    start_idx = 0
    map_idxes = []
    for num in cfg.model.model.get("out_channels"):
        end_idx = start_idx + num
        map_idxes.append([start_idx, end_idx])
        start_idx = end_idx

    # process and save
    cnt = 0
    map_names = ("albedo", "normal", "roughness", "metalness")
    for outs in predictions:
        names_batch, inputs_batch, gts_batch, preds_batch = outs

        # -1,1 -> 0,1  N,C,H,W -> N,H,W,C
        inputs_batch = torch.permute(torch.clamp(inputs_batch * 0.5 + 0.5, 0.0, 1.0), [0, 2, 3, 1])
        gts_batch = torch.permute(torch.clamp(gts_batch * 0.5 + 0.5, 0.0, 1.0), [0, 2, 3, 1])
        preds_batch = torch.permute(torch.clamp(preds_batch * 0.5 + 0.5, 0.0, 1.0), [0, 2, 3, 1])

        for n in range(len(names_batch)):
            name = names_batch[n]
            input_render = inputs_batch[n]
            gt_maps = gts_batch[n]
            pred_maps = preds_batch[n]

            # sample directory
            sample_out_dir = os.path.join(out_dir, f"{cnt}")
            os.makedirs(sample_out_dir)

            # name
            with open(os.path.join(sample_out_dir, "name.txt"), "w") as f:
                f.write(name)

            # input
            input_render = input_render.cpu().numpy() * 255.0
            input_render = Image.fromarray(input_render.astype(np.uint8))
            input_render.save(os.path.join(sample_out_dir, "input.jpg"))

            # gts
            gt_maps = gt_maps.cpu().numpy() * 255.0
            for idx, (start_idx, end_idx) in enumerate(map_idxes):
                one_map = gt_maps[..., start_idx:end_idx].astype(np.uint8)
                one_map = Image.fromarray(
                    one_map.repeat(3, axis=-1) if one_map.shape[-1] == 1 else one_map
                )
                one_map.save(os.path.join(sample_out_dir, f"{map_names[idx]}_gt.jpg"))

            # preds
            pred_maps = pred_maps.cpu().numpy() * 255.0
            for idx, (start_idx, end_idx) in enumerate(map_idxes):
                one_map = pred_maps[..., start_idx:end_idx].astype(np.uint8)
                one_map = Image.fromarray(
                    one_map.repeat(3, axis=-1) if one_map.shape[-1] == 1 else one_map
                )
                one_map.save(os.path.join(sample_out_dir, f"{map_names[idx]}_pred.jpg"))

            cnt += 1
            print(f"finished {cnt}")


if __name__ == "__main__":
    main()
