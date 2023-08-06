from pathlib import Path

import pytest
import torch
import hydra

from omegaconf import DictConfig, OmegaConf
from src.data.pbr_datamodule import PBRDataModule


@pytest.mark.parametrize("batch_size", [4, 16])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `PBRDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """

    cfg = OmegaConf.load("configs/data/test_pbr_datamodule.yaml")
    dm = hydra.utils.instantiate(cfg, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train
    assert Path(cfg.data_train.data_root_dir).exists()
    assert Path(cfg.data_train.data_list_file_dir).exists()
    for name in cfg.data_train.pbr_dir_names:
        assert Path(cfg.data_train.data_root_dir, name).exists()
    for name in cfg.data_train.rendered_dir_names:
        assert Path(cfg.data_train.data_root_dir, name).exists()

    dm.setup()
    assert dm.data_train
    assert isinstance(dm.train_dataloader(), torch.utils.data.DataLoader)

    num_datapoints = len(dm.data_train)
    assert num_datapoints == 200

    batch = next(iter(dm.train_dataloader()))
    _, input, gt = batch
    assert len(input) == batch_size
    assert len(gt) == batch_size
    assert input.shape == (batch_size, 3, 512, 512)
    assert gt.shape == (batch_size, 8, 512, 512)
    assert input.dtype == torch.float32
    assert gt.dtype == torch.float32