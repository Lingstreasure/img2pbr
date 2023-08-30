import torch
import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_b2


class EfficientNetModel(nn.Module):
    """EfficientNet model.

    The model consists of a pre-trained EfficientNetV2 trained on ImageNet-1K and a classifier.
    """

    def __init__(self, num_classes: int) -> None:
        """Initialize a `EfficientNetModel`.

        :param num_classes: The number of output class.
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone = efficientnet_b2()
        del self.backbone.classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True),
            nn.Linear(in_features=1408, out_features=num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images to specific classes."""
        x = self.backbone(x)
        return x
