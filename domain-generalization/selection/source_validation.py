from __future__ import annotations

from evaluation.domain_metrics import classification_metrics, predict


def evaluate_source_domains(model, val_datasets, device, num_workers=0, return_features=False):
    metrics_by_domain, outputs = {}, {}
    for domain, dataset in val_datasets.items():
        output = predict(model, dataset, device, num_workers=num_workers, return_features=return_features)
        metrics_by_domain[domain] = classification_metrics(output["labels"], output["predictions"])
        outputs[domain] = output
    return metrics_by_domain, outputs


def aggregate_source_metrics(metrics_by_domain):
    accuracies = [metrics["accuracy"] for metrics in metrics_by_domain.values()]
    macro_f1s = [metrics["macro_f1"] for metrics in metrics_by_domain.values()]
    return {
        "mean_source_accuracy": sum(accuracies) / len(accuracies),
        "worst_source_accuracy": min(accuracies),
        "mean_source_macro_f1": sum(macro_f1s) / len(macro_f1s),
        "worst_source_macro_f1": min(macro_f1s),
    }

