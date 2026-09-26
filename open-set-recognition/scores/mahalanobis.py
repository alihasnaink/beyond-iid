from __future__ import annotations

import torch


class DiagonalMahalanobis:
    """Nearest class-conditional mean under one shared diagonal covariance."""

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.means = None
        self.variance = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, num_classes: int = 10):
        features, labels = features.float(), labels.long()
        self.means = torch.stack([features[labels == c].mean(0) for c in range(num_classes)])
        residuals = features - self.means[labels]
        self.variance = residuals.square().mean(0) + self.epsilon
        return self

    def score(self, features: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
        if self.means is None or self.variance is None:
            raise RuntimeError("Call fit before score.")
        values = []
        for chunk in features.float().split(chunk_size):
            distances = ((chunk[:, None, :] - self.means[None, :, :]).square() /
                         self.variance[None, None, :]).sum(2)
            values.append(distances.amin(1))
        return torch.cat(values)

    def state_dict(self):
        return {"means": self.means, "variance": self.variance, "epsilon": self.epsilon}
