from typing import Dict, Tuple

import torch
from torch import nn

from src.models.components.losses.lpips import LPIPS


class PBRReconstructionLoss(nn.Module):
    """The PBR maps reconstruction losses module.
    
    The loss module contain reconstruction loss, perceptual loss, render loss and GAN-based losses. 
    """
    
    def __init__(
        self,
        random_per_loss_weight: float = 0., 
        per_loss_weight: float = 0., 
        render_loss_weight: float = 0., 
        gan_loss_weight: float = 0.
    ) -> None:
        """
        :param random_per_loss_weight: The weight of channels-random-selected perceptual loss. Default to `0.`.
        :param per_loss_weight: The weight of perceptual loss. Default to `0.`.
        :param render_loss_weight: The weight of differentialable rendering loss. Default to `0.`.
        :param gan_loss_weight: The weight of GAN-based loss. Default to `0.`.
        """
        super().__init__()
        
        if random_per_loss_weight > 0. or per_loss_weight > 0.:
            self.perceptual_loss = LPIPS().eval()
            ## using different layer weights of VGG19
            # layer_weights={"conv1_2": 0.1, "conv2_2": 0.1, "conv3_4": 1., "conv4_4": 1., "conv5_4": 1.}
            # self.perceptual_loss = basicsr.losses.basic_loss.PerceptualLoss(layer_weights=layer_weights)
            
        if render_loss_weight > 0.:
            # self.renderer = Renderer(normal_format='dx')
            pass
            
        self.random_per_loss_weight = random_per_loss_weight
        self.per_loss_weight = per_loss_weight
        self.render_loss_weight = render_loss_weight
        self.gan_loss_weight = gan_loss_weight

    def forward(self, preds: torch.Tensor, gts: torch.Tensor) -> Tuple[float, Dict[str, torch.Tensor]]:
        B, C, H, W = preds.shape
        
        # pixel-wised reconstruction loss - the base loss
        rec_loss = torch.abs(preds.contiguous() - gts.contiguous()).mean()
        
        # logger dict for each loss
        log = {"rec_loss": rec_loss.detach()}
        
        # perceptual loss
        p_loss = 0.
        if self.per_loss_weight > 0.:
            p_loss += self.perceptual_loss(preds[:, :3, ...].contiguous(), gts[:, :3, ...].contiguous())[0].mean()
            p_loss += self.perceptual_loss(preds[:, 3:6, ...].contiguous(), gts[:, 3:6, ...].contiguous())[0].mean()
            p_loss += self.perceptual_loss(preds[:, 6:7, ...].repeat([1, 3, 1, 1]).contiguous(), gts[:, 6:7, ...].repeat([1, 3, 1, 1]).contiguous())[0].mean()
            p_loss += self.perceptual_loss(preds[:, 7:8, ...].repeat([1, 3, 1, 1]).contiguous(), gts[:, 7:8, ...].repeat([1, 3, 1, 1]).contiguous()).mean()
            p_loss *= self.per_loss_weight
            log["per_loss"] = p_loss.detach()
        
        # differentiable rendering loss
        render_loss = 0.
        if self.render_loss_weight > 0.:
            if self.renderer.device != gts.device:
                self.renderer.to(gts.device)
            render_imgs = self.renderer.evaluate(*[(preds[:, :3, ...] + 1.0) / 2.0, 
                                                   (preds[:, 3:6, ...] + 1.0) / 2.0, 
                                                   (preds[:, 6:7, ...] + 1.0) / 2.0, 
                                                   (preds[:, 7:, ...] + 1.0) / 2.0, 
                                                   torch.ones((B, 1, H, W), device=gts.device).float()])
            render_loss = torch.abs((preds * 2.0 - 1).contiguous() - render_imgs.contiguous()).mean() * self.render_weight
            log["render_loss"] = render_loss.detach()

        # channels-random-selected perceptual loss
        random_per_loss = 0.
        if self.random_per_loss_weight > 0.:
            indexes = torch.randint(0, C, (3,))  # random select 3 channels
            random_per_loss = self.perceptual_loss(preds[:, indexes, ...].contiguous(), 
                                                 gts[:, indexes, ...].contiguous()).mean() * self.random_per_loss_weight
            log["random_per_loss"] = random_per_loss.detach()

        # total loss
        loss = rec_loss + random_per_loss + render_loss + p_loss

        return loss, log
