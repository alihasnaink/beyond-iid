import torch


def msp_unknownness(logits: torch.Tensor) -> torch.Tensor:
    """Larger values indicate that an example is more likely to be unknown."""
    return 1.0 - logits.softmax(dim=1).amax(dim=1)
