from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def source_domain_separability(features_by_domain, seed=6304):
    """Balanced three-source probe with a seeded 70/30 split and C=1."""
    rng = np.random.default_rng(seed)
    domains = list(features_by_domain)
    count = min(len(features_by_domain[domain]) for domain in domains)
    feature_parts, label_parts = [], []
    for domain_index, domain in enumerate(domains):
        features = np.asarray(features_by_domain[domain])
        chosen = rng.choice(len(features), count, replace=False)
        feature_parts.append(features[chosen])
        label_parts.append(np.full(count, domain_index, dtype=int))
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )
    probe.fit(train_x, train_y)
    return accuracy_score(test_y, probe.predict(test_x)), count

