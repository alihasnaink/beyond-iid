from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        network = resnet18(weights=None, num_classes=num_classes)
        network.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        network.maxpool = nn.Identity()
        self.network = network

    def forward_to_layer2(self, x):
        n = self.network
        x = n.relu(n.bn1(n.conv1(x)))
        x = n.maxpool(x)
        x = n.layer1(x)
        return n.layer2(x)

    def features_from_layer2(self, x):
        n = self.network
        x = n.layer3(x)
        x = n.layer4(x)
        return torch.flatten(n.avgpool(x), 1)

    def extract_features(self, x):
        return self.features_from_layer2(self.forward_to_layer2(x))

    def known_logits_from_features(self, features):
        return self.network.fc(features)

    def forward(self, x, return_features=False):
        features = self.extract_features(x)
        logits = self.known_logits_from_features(features)
        return (logits, features) if return_features else logits


class PROSERResNet18(CIFARResNet18):
    def __init__(self, num_classes=10, dummy_classifiers=5):
        super().__init__(num_classes)
        self.dummy_classifier = nn.Linear(512, dummy_classifiers)

    def logits_from_features(self, features):
        return self.known_logits_from_features(features), self.dummy_classifier(features)

    def forward(self, x, return_features=False, return_dummy=False):
        features = self.extract_features(x)
        known, dummy = self.logits_from_features(features)
        if return_dummy and return_features:
            return known, dummy, features
        if return_dummy:
            return known, dummy
        return (known, features) if return_features else known
