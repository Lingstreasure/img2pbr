from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.rmse_sw import RootMeanSquaredErrorUsingSlidingWindow
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


class IMG2PBRGANLitModule(LightningModule):
    """`IMG2PBRGANLitModule`for SVBRDF reconstruction.

    A GAN based lightning module.

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
        adv_loss_weight: float,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        loss_G: torch.nn.Module,
        loss_D: torch.nn.Module,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        scheduler_G: torch.optim.lr_scheduler = None,
        scheduler_D: torch.optim.lr_scheduler = None,
    ) -> None:
        """Initialize a `IMG2PBRGANLitModule`.

        :param adv_loss_weight: The weight of adversarial loss.
        :param generator: The generator model for generating pbr maps.
        :param discriminator: The discriminator model for judging whether real/fake of generated
            maps.
        :param loss_G: The generator loss to calculate on generated pbr maps.
        :param loss_D: The discriminator loss to calculate on judging real/fake pbr maps.
        :param optimizer_G: The optimizer used for training generator.
        :param optimizer_D: The optimizer used for training discriminator.
        :param scheduler_G: The learning rate scheduler used for generator.
        :param scheduler_G: The learning rate scheduler used for discriminator.
        """
        super().__init__()
        # lightning 2.x doesn't support automatic optimization
        self.automatic_optimization = False

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["generator", "discriminator", "loss_G", "loss_D", "adv_loss_weight"],
        )

        # generator and discriminator
        self.G = generator
        self.D = discriminator

        # losses
        self.loss_G = loss_G
        self.loss_D = loss_D
        self.adv_loss_weight = adv_loss_weight

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Perform a forward pass through the generator model `self.G`.

        :param x: A tensor of images.
        :return: multi tensors of pbr maps.
        """
        return self.G(x)

    def _G_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """A generator training step.

        :param inputs: Tensor of input images.
        :param targets: Tensor of ground truth pbr maps corresponding to input images.
        :return: Losses of generator
        """
        # predict
        preds = self.G(inputs)
        disc_middle_logits, disc_pixel_logits = self.D(preds)

        # calculate adversarial loss
        adversarial_loss = self.loss_D(
            disc_middle_logits, torch.ones_like(disc_middle_logits)
        ) + self.loss_D(disc_pixel_logits.view(-1), torch.ones_like(disc_pixel_logits.view(-1)))
        adversarial_loss *= self.adv_loss_weight

        # calculate generator loss
        loss, loss_dict = self.loss_G(preds, targets, self.current_epoch)
        loss += adversarial_loss

        # log
        loss_dict["adversarial_loss"] = adversarial_loss.detach()

        return loss, loss_dict

    def _D_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """A discriminator training step."""
        # predict
        preds = self.G(inputs).detach()
        fake_middle_logits, fake_pixel_logits = self.D(preds)
        real_middle_logits, real_pixel_logits = self.D(targets)

        # calculate loss
        fake_loss = self.loss_D(
            fake_middle_logits, torch.zeros_like(fake_middle_logits)
        ) + self.loss_D(fake_pixel_logits.view(-1), torch.zeros_like(fake_pixel_logits.view(-1)))
        real_loss = self.loss_D(
            real_middle_logits, torch.ones_like(real_middle_logits)
        ) + self.loss_D(real_pixel_logits.view(-1), torch.ones_like(real_pixel_logits.view(-1)))

        # log
        loss_dict = {}
        loss_dict["fake_loss"] = fake_loss.detach()
        loss_dict["real_loss"] = real_loss.detach()

        return 0.5 * (fake_loss + real_loss), loss_dict

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def training_step(
        self,
        batch: Tuple[str, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the sample name and input, gt tensor of
            images.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        name, inputs, gts = batch
        opt_G, opt_D = self.optimizers()

        ################################
        #          Generator
        ################################
        opt_G.zero_grad()

        # calculate G loss
        G_loss, G_loss_dict = self._G_step(inputs, gts)

        # backward and update gards
        self.manual_backward(G_loss)
        opt_G.step()

        # log G loss
        self.log(
            "train/G/G_loss", G_loss.clone().detach(), prog_bar=True, on_step=True, on_epoch=True
        )
        for loss_k, loss_v in G_loss_dict.items():
            self.log(f"train/G/{loss_k}", loss_v, prog_bar=False, on_step=False, on_epoch=True)

        ################################
        #         Discriminator
        ################################
        opt_D.zero_grad()

        # calculate D loss
        D_loss, D_loss_dict = self._D_step(inputs, gts)

        # backward and update gards
        self.manual_backward(D_loss)
        opt_D.step()

        # log D loss
        self.log(
            "train/D/D_loss", D_loss.clone().detach(), prog_bar=True, on_step=True, on_epoch=True
        )
        for loss_k, lossv in D_loss_dict.items():
            self.log(f"train/D/{loss_k}", loss_v, prog_bar=False, on_step=False, on_epoch=True)

    def validation_step(
        self, batch: Tuple[str, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the sample name and input, gt tensor of
            images.
        :param batch_idx: The index of the current batch.
        """
        name, inputs, gts = batch

        # generator
        G_loss, G_loss_dict = self._G_step(inputs, gts)
        self.log(
            "val/G/G_loss", G_loss.clone().detach(), prog_bar=True, on_step=False, on_epoch=True
        )
        for loss_k, loss_v in G_loss_dict.items():
            self.log(f"val/G/{loss_k}", loss_v, prog_bar=False, on_step=False, on_epoch=True)

        # discriminator
        D_loss, D_loss_dict = self._D_step(inputs, gts)
        self.log(
            "val/D/D_loss", D_loss.clone().detach(), prog_bar=True, on_step=False, on_epoch=True
        )
        for loss_k, lossv in D_loss_dict.items():
            self.log(f"val/D/{loss_k}", loss_v, prog_bar=False, on_step=False, on_epoch=True)

    def on_test_start(self) -> None:
        """Lightning hook that is called when testing begins."""
        # metrics for multi pbr maps
        self.lpips_list = torch.nn.ModuleList()
        self.ssim_list = torch.nn.ModuleList()
        self.rmse_list = torch.nn.ModuleList()
        for i in range(len(self.G.out_channels)):
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
        preds = self(inputs)
        start_idx = 0
        for i, num in enumerate(self.G.out_channels):
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
        preds = self(inputs)
        return names, inputs, gts, preds

    def configure_optimizers(self) -> Tuple[List]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        :return: A dict containing the configured optimizers and learning-rate schedulers to be
            used for training.
        """
        optimizer_G = self.hparams.optimizer_G(params=self.G.parameters())
        optimizer_D = self.hparams.optimizer_D(params=self.D.parameters())
        if self.hparams.scheduler_G is not None or self.hparams.scheduler_D is not None:
            scheduler_G = self.hparams.scheduler_G(optimizer=optimizer_G)
            scheduler_D = self.hparams.scheduler_D(optimizer=optimizer_D)
            return [optimizer_G, optimizer_D], [
                {
                    "scheduler": scheduler_G,
                    "monitor": "val/G/rec_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
                {
                    "scheduler": scheduler_D,
                    "monitor": "val/G/rec_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            ]

        return [optimizer_G, optimizer_D], []

    @torch.no_grad()
    def log_images(self, batch: Tuple[str, torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Log model prediction images and gt images.

        :return: A dict of images to be logged.
        """
        log = dict()
        name, inputs, gts = batch
        inputs = inputs.to(self.device)
        preds = self(inputs)

        log["inputs"] = inputs
        _, C, _, _ = preds.shape
        if C > 3:
            map_type = ["albedo", "normal", "rough", "metal"]
            start_idx = 0
            for idx, num in enumerate(self.G.out_channels):
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
