import torch.nn as nn


def make_classifier_head(feature_dim=512, num_classes=7):
    return nn.Linear(feature_dim, num_classes)
