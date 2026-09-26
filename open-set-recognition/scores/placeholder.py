import torch


def placeholder_unknownness(known_logits: torch.Tensor, dummy_logits: torch.Tensor) -> torch.Tensor:
    """PROSER decision margin: strongest dummy minus strongest known classifier."""
    return dummy_logits.amax(1) - known_logits.amax(1)


def calibrate_placeholder(validation_scores: torch.Tensor, quantile: float = 0.95):
    """Return the additive bias that makes validation's 95th percentile equal zero."""
    return -torch.quantile(validation_scores.float(), quantile).item()
