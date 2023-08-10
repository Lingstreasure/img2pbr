from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Tuple

import torch
from torch import nn

from src.models.components.losses.lpips import LPIPS


class PBRReconstructionLoss(nn.Module):
    """The PBR maps reconstruction losses module.

    The loss module contain reconstruction loss, perceptual loss, render loss and GAN-based losses.
    """

    def __init__(
        self,
        rec_loss_weights: List[float],
        pbr_ch_nums: List[int],
        per_loss_net: Literal["alex", "vgg"] = "alex",
        per_loss_after_num_epochs: int = 10,
        random_per_loss_weight: float = 0.0,
        per_loss_weight: float = 0.0,
        render_loss_weight: float = 0.0,
        gan_loss_weight: float = 0.0,
    ) -> None:
        """
        :param rec_loss_weights: The list of reconstruction loss weights between pbr maps, equal with numbers of model decoders.
        :param pbr_ch_nums: The list contained channel numbers of multi pbr maps, equal with numbers of model decoders.
        :param per_loss_net: The pretrained net used for calculating LPIPS. Default to `alex`.
        :param per_loss_after_num_epochs: Activate the perceptual loss after the number epochs. Default to `10`.
        :param random_per_loss_weight: The weight of channels-random-selected perceptual loss. Default to `0.`.
        :param per_loss_weight: The weight of perceptual loss. Default to `0.`.
        :param render_loss_weight: The weight of differentialable rendering loss. Default to `0.`.
        :param gan_loss_weight: The weight of GAN-based loss. Default to `0.`.
        """
        super().__init__()
        assert len(rec_loss_weights) == len(
            pbr_ch_nums
        ), "rec loss weights list and should own same length with pbr map numbers"

        self.rec_loss_weights = rec_loss_weights
        self.pbr_ch_nums = pbr_ch_nums

        if random_per_loss_weight > 0.0 or per_loss_weight > 0.0:
            self.perceptual_loss = LPIPS(net=per_loss_net).eval()

        if render_loss_weight > 0.0:
            # self.renderer = Renderer(normal_format='dx')
            pass

        self.p_loss_after_num_epochs = per_loss_after_num_epochs
        self.random_per_loss_weight = random_per_loss_weight
        self.per_loss_weight = per_loss_weight
        self.render_loss_weight = render_loss_weight
        self.gan_loss_weight = gan_loss_weight

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, cur_epoch: int
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Cal the losses between preds and targets."""
        B, C, H, W = preds.shape
        log = {}  # logger dict for each loss

        # pixel-wised reconstruction loss, calculate each map loss separately - the base loss
        rec_loss = 0.0
        start_idx = 0
        for idx, num in enumerate(self.pbr_ch_nums):
            end_idx = start_idx + num
            single_map_loss = torch.abs(
                preds[:, start_idx:end_idx, ...].contiguous()
                - targets[:, start_idx:end_idx, ...].contiguous()
            ).mean()
            rec_loss += single_map_loss * self.rec_loss_weights[idx]
            log[f"rec_loss/{idx}"] = single_map_loss.detach()
            start_idx = end_idx

        log["rec_loss"] = rec_loss.detach()

        # perceptual loss, only start when weight and epoch conditions are met
        p_loss = 0.0
        if self.per_loss_weight > 0.0 and cur_epoch > self.p_loss_after_num_epochs:
            start_idx = 0
            for num in self.pbr_ch_nums:
                end_idx = start_idx + num
                p_loss += self.perceptual_loss(
                    preds[:, start_idx:end_idx, ...].contiguous(),
                    targets[:, start_idx:end_idx, ...].contiguous(),
                ).mean()
                start_idx = end_idx

            p_loss *= self.per_loss_weight
            log["per_loss"] = p_loss.detach()

        # channels-random-selected perceptual loss, same conditions as above
        random_per_loss = 0.0
        if self.random_per_loss_weight > 0.0 and cur_epoch > self.p_loss_after_num_epochs:
            indexes = torch.randint(0, C, (3,))  # random select 3 channels
            random_per_loss = self.perceptual_loss(
                preds[:, indexes, ...].contiguous(), targets[:, indexes, ...].contiguous()
            ).mean()
            random_per_loss = random_per_loss * self.random_per_loss_weight
            log["random_per_loss"] = random_per_loss.detach()

        # differentiable rendering loss
        render_loss = 0.0
        if self.render_loss_weight > 0.0:
            if self.renderer.device != targets.device:
                self.renderer.to(targets.device)
            render_imgs = self.renderer.evaluate(
                *[
                    (preds[:, :3, ...] + 1.0) / 2.0,
                    (preds[:, 3:6, ...] + 1.0) / 2.0,
                    (preds[:, 6:7, ...] + 1.0) / 2.0,
                    (preds[:, 7:, ...] + 1.0) / 2.0,
                    torch.ones((B, 1, H, W), device=targets.device).float(),
                ]
            )
            render_loss = (
                torch.abs((preds * 2.0 - 1).contiguous() - render_imgs.contiguous()).mean()
                * self.render_weight
            )
            log["render_loss"] = render_loss.detach()

        # total loss
        loss = rec_loss + random_per_loss + render_loss + p_loss

        return loss, log
