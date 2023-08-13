from typing import Callable, Dict, List, Literal, Tuple

import torch
import torch.nn as nn
from focal_frequency_loss import FocalFrequencyLoss as FFL

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
        per_loss_after_num_epochs: int = -1,
        random_per_loss_weight: float = 0.0,
        per_loss_weight: float = 0.0,
        focal_loss_weight: float = 0.0,
        render_loss_weight: float = 0.0,
        gan_loss_weight: float = 0.0,
    ) -> None:
        """
        :param rec_loss_weights: The list of reconstruction loss weights between pbr maps, equal with numbers of model decoders.
        :param pbr_ch_nums: The list contained channel numbers of multi pbr maps, equal with numbers of model decoders.
        :param per_loss_net: The pretrained net used for calculating LPIPS. Default to `alex`.
        :param per_loss_after_num_epochs: Activate the perceptual loss after the number epochs. Default to `-1`.
        :param random_per_loss_weight: The weight of channels-random-selected perceptual loss. Default to `0.`.
        :param per_loss_weight: The weight of perceptual loss. Default to `0.`.
        :param focal_loss_weight: The weight of focal frequency loss. Default to `0`.
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

        if focal_loss_weight > 0.0:
            self.ff_loss = FFL(loss_weight=focal_loss_weight)

        self.p_loss_after_num_epochs = per_loss_after_num_epochs
        self.random_per_loss_weight = random_per_loss_weight
        self.per_loss_weight = per_loss_weight
        self.render_loss_weight = render_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.gan_loss_weight = gan_loss_weight

        self.map_idxes = []
        start_idx = 0
        for num in self.pbr_ch_nums:
            end_idx = start_idx + num
            self.map_idxes.append((start_idx, end_idx))
            start_idx = end_idx

    def multi_map_wrapper(loss_func: Callable) -> Callable:
        """Decorator that supply indexes for calculating losses of multi pbr maps.

        :param loss_func: The built-in loss function in this class to be wrapped.
        :return: The wrapped loss function.
        """

        def wrap(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Calculate multi map losses.

            :param pred: Predicted tensor of multi pbr maps.
            :param target: Ground truth tensor of multi pbr maps.
            """
            losses = []
            for start, end in self.map_idxes:
                losses.append(loss_func(self, pred[:, start:end, :], target[:, start:end, :]))
            return losses

        return wrap

    @multi_map_wrapper
    def cal_focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate focal frequency loss on a pbr map."""
        return self.ff_loss(pred.contiguous(), target.contiguous())

    @multi_map_wrapper
    def cal_rec_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss on a pbr map."""
        return torch.abs(pred.contiguous() - target.contiguous()).mean()

    @multi_map_wrapper
    def cal_per_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual loss on a pbr map."""
        if target.shape[1] == 1:
            pred = pred.repeat([1, 3, 1, 1])
            target = target.repeat([1, 3, 1, 1])
        return self.perceptual_loss(pred.contiguous(), target.contiguous()).mean()

    def cal_random_per_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate random perceptual loss on multi pbr maps.

        This loss function calculate perceptual loss on 3 random channels in input.
        """
        indexes = torch.randint(0, target.shape[1], (3,))  # random select 3 channels
        return self.perceptual_loss(
            pred[:, indexes, ...].contiguous(), target[:, indexes, ...].contiguous()
        ).mean()

    def cal_render_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate differentialle rendering loss on all pbr maps.

        :param pred: The predicted tensor with all pbr maps concatenated in C dimension.
        :param target: The ground truth tensor with all pbr maps concatenated in C dimension.
        """
        B, _, H, W = target.shape
        if self.renderer.device != target.device:
            self.renderer.to(target.device)
        pred_render_imgs = self.renderer.evaluate(
            *[
                (pred[:, :3, ...] + 1.0) / 2.0,
                (pred[:, 3:6, ...] + 1.0) / 2.0,
                (pred[:, 6:7, ...] + 1.0) / 2.0,
                (pred[:, 7:, ...] + 1.0) / 2.0,
                torch.ones((B, 1, H, W), device=target.device).float(),
            ]
        )
        target_render_imgs = self.renderer.evaluate(
            *[
                (target[:, :3, ...] + 1.0) / 2.0,
                (target[:, 3:6, ...] + 1.0) / 2.0,
                (target[:, 6:7, ...] + 1.0) / 2.0,
                (target[:, 7:, ...] + 1.0) / 2.0,
                torch.ones((B, 1, H, W), device=target.device).float(),
            ]
        )
        return torch.abs(pred_render_imgs.contiguous() - target_render_imgs.contiguous()).mean()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, cur_epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate the losses between pred and target of multi pbr maps.

        :param pred: The predicted tensor with all pbr maps concatenated in C dimension.
        :param target: The ground truth tensor with all pbr maps concatenated in C dimension.
        """
        log = {}  # logger dict for each loss

        # pixel-wised reconstruction loss, calculate each map loss separately - the base loss
        rec_loss = 0.0
        losses = self.cal_rec_loss(pred, target)
        for idx, loss in enumerate(losses):
            rec_loss += loss * self.rec_loss_weights[idx]
            log[f"rec_loss/{idx}"] = loss.detach()

        log["rec_loss"] = rec_loss.detach()

        # perceptual loss, only start when weight and epoch conditions are met
        p_loss = 0.0
        if self.per_loss_weight > 0.0 and cur_epoch > self.p_loss_after_num_epochs:
            p_loss += sum(self.cal_per_loss(pred, target))
            p_loss *= self.per_loss_weight
            log["per_loss"] = p_loss.detach()

        # channels-random-selected perceptual loss, same conditions as above
        random_per_loss = 0.0
        if self.random_per_loss_weight > 0.0 and cur_epoch > self.p_loss_after_num_epochs:
            random_per_loss = self.cal_random_per_loss(pred, target)
            random_per_loss *= self.random_per_loss_weight
            log["random_per_loss"] = random_per_loss.detach()

        # focal frequency loss
        ff_loss = 0.0
        if self.focal_loss_weight > 0.0:
            ff_loss = sum(self.cal_focal_loss(pred, target))
            ff_loss *= self.focal_loss_weight
            log["ff_loss"] = ff_loss.detach()

        # differentiable rendering loss
        render_loss = 0.0
        if self.render_loss_weight > 0.0:
            render_loss = self.cal_render_loss(pred, target)
            render_loss *= self.render_loss_weight
            log["render_loss"] = render_loss.detach()

        # total loss
        loss = rec_loss + p_loss + random_per_loss + ff_loss + render_loss

        return loss, log
