from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class PACSClassifier(nn.Module):
    """Architecture-identical to the shared Task 2 ERM model."""

    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, images, return_features=False):
        features = self.backbone(images).flatten(1)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def build_model(num_classes=7, pretrained=True):
    return PACSClassifier(num_classes, pretrained=pretrained)


def freeze_batchnorm_stats(module):
    """Keep ImageNet running statistics while gamma and beta remain trainable."""
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()

