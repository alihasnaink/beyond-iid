from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class PACSClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, images, return_features=False):
        features = self.backbone(images).flatten(1)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def build_model(num_classes=7):
    return PACSClassifier(num_classes)


def freeze_batchnorm_stats(module):
    """Freeze running statistics while leaving affine gamma/beta trainable."""
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()
