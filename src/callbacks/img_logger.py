import os
from typing import Any, Dict

import numpy as np
import torch, torchvision, lightning
from PIL import Image
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only 


class ImageLogger(Callback):
    """Log images from lightning module output.
    
    This logger should be used with the function log_image() implemented in Lightning Module.
    """
    
    def __init__(
        self, 
        check_batch_idx: int, 
        max_images: int = 4, 
        clamp: bool = True, 
        rescale: bool = True, 
        log_first_step: bool = True,
        log_all_val: bool = True
    ) -> None:
        """Initilize the ImageLogger.
        
        :param check_batch_idx: The batch_idx for checking whether to log image.
        :param max_images: The max number to be logged in one batch. Default to `4`.
        :param clamp: Whether to clamp the image tensor to [-1., 1.]. Default to `True`.
        :param rescale: Whether to rescale the tensor to [0., 1.] when saving images. Default to `True`.
        :param log_first_step: Whether to log first step of training and validating. Default to `True`.
        :param log_all_val: Whether to log all validation steps. Default to `True`.
        """
        super().__init__()
        
        self.rescale = rescale
        self.check_batch_idx = check_batch_idx
        self.max_images = max_images
        self.clamp = clamp
        self.log_first_step = log_first_step
        self.log_all_val = log_all_val

    @rank_zero_only
    def _logging(
        self, 
        pl_module: lightning.LightningModule, 
        images: Dict[str, torch.Tensor], 
        split: str
    ) -> None:
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(
        self, 
        save_dir: str,
        split: str, 
        images: Dict[str, torch.Tensor],
        global_step: int, 
        current_epoch: int, 
        batch_idx: int
    ) -> None:
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=images[k].shape[0])
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.permute(1, 2, 0).squeeze(-1)  # h w c
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:05}_e-{:03}_b-{:04}.jpg".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(
        self, 
        pl_module: lightning.LightningModule, 
        batch: Any, 
        batch_idx: int, 
        split="train"
    ) -> None:
        if self.log_all_val and split == "val":
            should_log = True
        else:
            should_log = self.check_frequency(batch_idx)
        if (should_log and  (batch_idx == self.check_batch_idx) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
                        
            self._logging(pl_module, images, split)
                
            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx: int) -> bool:
        if check_idx == self.check_batch_idx and (
                check_idx > 0 or self.log_first_step):
            return True
        return False

    def on_train_batch_end(
        self, 
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule, 
        outputs: Any,
        batch: Any, 
        batch_idx: int, 
    ) -> None:
        if pl_module.global_step > 0 or self.log_first_step:
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, 
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule, 
        outputs: Any,
        batch: Any, 
        batch_idx: int, 
    ) -> None:
        if pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")