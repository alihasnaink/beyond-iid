import torch.nn.functional as F


def vanilla_loss(logits, labels):
    return F.cross_entropy(logits, labels)
