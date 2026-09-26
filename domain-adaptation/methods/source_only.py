import torch.nn.functional as F


def source_only_loss(source_logits, source_labels):
    return F.cross_entropy(source_logits, source_labels)
