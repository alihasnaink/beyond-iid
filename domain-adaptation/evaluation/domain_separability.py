from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def domain_separability(source_features, target_features, seed=6304):
    """Balanced source-vs-target probe with a seeded 70/30 split and C=1."""
    rng = np.random.default_rng(seed)
    count = min(len(source_features), len(target_features))
    source_idx = rng.choice(len(source_features), count, replace=False)
    target_idx = rng.choice(len(target_features), count, replace=False)
    features = np.concatenate([source_features[source_idx], target_features[target_idx]])
    labels = np.concatenate([np.zeros(count, dtype=int), np.ones(count, dtype=int)])
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.30, random_state=seed, stratify=labels,
    )
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    probe.fit(train_x, train_y)
    return accuracy_score(test_y, probe.predict(test_x)), count
