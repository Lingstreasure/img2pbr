from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.rmse_sw import RootMeanSquaredErrorUsingSlidingWindow
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


class IMG2PBRLitModule(LightningModule):
    """`LightningModule`for SVBRDF reconstruction.

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
    https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        """Initialize a `IMG2PBRLitModule`.

        :param model: The model to train.
        :param loss: The loss to calculate on reconstructed pbr maps.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["model", "loss"])

        # model
        self.model = model

        # loss
        self.loss = loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Perform a forward pass through the model `self.model`.

        :param x: A tensor of images.
        :return: multi tensors of pbr maps.
        """
        return self.model(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(
        self, batch: Tuple[str, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the sample name and input, gt tensor of images.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
        """
        name, inputs, gts = batch
        preds = self.forward(inputs)
        loss, loss_dict = self.loss(preds, gts, self.current_epoch)
        return loss, loss_dict, preds, gts

    def training_step(
        self, batch: Tuple[str, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the sample name and input, gt tensor of
            images.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, loss_dict, preds, targets = self.model_step(batch)

        # update and log metrics
        self.log("train/loss", loss.clone().detach(), prog_bar=True, on_step=True, on_epoch=True)
        for loss_k, loss_v in loss_dict.items():
            self.log(f"train/{loss_k}", loss_v, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[str, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the sample name and input, gt tensor of
            images.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_dict, preds, targets = self.model_step(batch)

        # update and log metrics
        self.log("val/loss", loss.clone().detach(), prog_bar=True, on_step=True, on_epoch=True)
        for loss_k, loss_v in loss_dict.items():
            self.log(f"val/{loss_k}", loss_v, prog_bar=False, on_step=False, on_epoch=True)

    # def on_validation_epoch_end(self) -> None:
    #     "Lightning hook that is called when a validation epoch ends."
    #     lpips = self.lpips.compute()
    #     ssim = self.ssim.compute()
    #     rmse = self.rmse.compute()

    #     self.log("val/lpips", lpips, sync_dist=True, prog_bar=False)
    #     self.log("val/ssim", ssim, sync_dist=True, prog_bar=False)
    #     self.log("val/rmse", rmse, sync_dist=True, prog_bar=False)

    def on_test_start(self) -> None:
        """Lightning hook that is called when testing begins."""
        # metrics for multi pbr maps
        self.lpips_list = torch.nn.ModuleList()
        self.ssim_list = torch.nn.ModuleList()
        self.rmse_list = torch.nn.ModuleList()
        for i in range(len(self.model.out_channels)):
            self.lpips_list.append(LearnedPerceptualImagePatchSimilarity().to(self.device))
            self.ssim_list.append(StructuralSimilarityIndexMeasure().to(self.device))
            self.rmse_list.append(RootMeanSquaredErrorUsingSlidingWindow().to(self.device))

    def test_step(self, batch: Tuple[str, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the sample name and input, gt tensor of
            images.
        :param batch_idx: The index of the current batch.
        """

        name, inputs, gts = batch
        preds = self.forward(inputs)
        start_idx = 0
        for i, num in enumerate(self.model.out_channels):
            end_idx = start_idx + num

            # extract each pbr map
            pred = preds[:, start_idx:end_idx, ...]
            target = gts[:, start_idx:end_idx, ...]
            if pred.shape[1] == 1:
                pred = pred.repeat([1, 3, 1, 1])
                target = target.repeat([1, 3, 1, 1])

            # update and log metrics
            self.lpips_list[i](pred, target)
            self.ssim_list[i](pred, target)
            self.rmse_list[i](pred, target)

            self.log(
                f"test/lpips/{i}", self.lpips_list[i], prog_bar=False, on_step=False, on_epoch=True
            )
            self.log(
                f"test/ssim/{i}", self.ssim_list[i], prog_bar=False, on_step=False, on_epoch=True
            )
            self.log(
                f"test/rmse/{i}", self.rmse_list[i], prog_bar=False, on_step=False, on_epoch=True
            )

            start_idx = end_idx

    def predict_step(
        self, batch: Tuple[str, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[Any]:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the sample name and input, gt tensor of
            images.
        :param batch_idx: The index of the current batch.
        """

        names, inputs, gts = batch
        preds = self.model(inputs)
        return names, inputs, gts, preds

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        :return: A dict containing the configured optimizers and learning-rate schedulers to be
            used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/rec_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @torch.no_grad()
    def log_images(self, batch: Tuple[str, torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """A interface of ImageLogger callback for logging images.

        Log model prediction images and gt images.
        :return: A dict of images to be logged.
        """
        log = dict()
        name, inputs, gts = batch
        inputs = inputs.to(self.device)
        preds = self.forward(inputs)

        log["inputs"] = inputs
        _, C, _, _ = preds.shape
        if C > 3:
            map_type = ["albedo", "normal", "rough", "metal"]
            start_idx = 0
            for idx, num in enumerate(self.model.out_channels):
                end_idx = start_idx + num

                # extract each pbr map
                pred = preds[:, start_idx:end_idx, ...]
                target = gts[:, start_idx:end_idx, ...]

                if pred.size(1) == 1:  # rough or metal
                    pred = pred.repeat(1, 3, 1, 1)
                    target = target.repeat(1, 3, 1, 1)

                # log the pbr map
                log[map_type[idx] + "_rec"] = pred
                log[map_type[idx]] = target

                start_idx = end_idx

        return log
