import torch


def mls_unknownness(logits: torch.Tensor) -> torch.Tensor:
    """Negative maximum logit, oriented so larger values mean more unknown."""
    return -logits.amax(dim=1)
