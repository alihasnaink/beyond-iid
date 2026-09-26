from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


@torch.inference_mode()
def predict(model, dataset, device, batch_size=64, num_workers=2, return_features=False):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    labels, predictions, probabilities, features, identifiers = [], [], [], [], []
    for images, target, item_ids in loader:
        logits, batch_features = model(images.to(device, non_blocking=True), return_features=True)
        probs = logits.softmax(1).cpu()
        labels.extend(torch.as_tensor(target).tolist())
        predictions.extend(probs.argmax(1).tolist())
        probabilities.append(probs)
        if return_features:
            features.append(batch_features.float().cpu())
        identifiers.extend(list(item_ids))
    output = {
        "labels": np.asarray(labels), "predictions": np.asarray(predictions),
        "probabilities": torch.cat(probabilities), "identifiers": identifiers,
    }
    if return_features:
        output["features"] = torch.cat(features)
    return output


def classification_metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def evaluate_domains(model, datasets, device, batch_size=64, num_workers=2):
    results = {}
    for domain, dataset in datasets.items():
        output = predict(model, dataset, device, batch_size, num_workers)
        results[domain] = classification_metrics(output["labels"], output["predictions"])
    return results
