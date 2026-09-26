from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score


def metric_row(model, condition, probabilities, labels, clean_predictions=None):
    predictions = probabilities.argmax(1)
    row = {
        "model": model,
        "condition": condition,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "mean_max_confidence": probabilities.max(1).values.mean().item(),
        "prediction_consistency": np.nan if clean_predictions is None else predictions.eq(clean_predictions).float().mean().item(),
    }
    return row, predictions


def evaluate_standard(all_models, feature_cache, labels, probability_fn, results_dir):
    rows, predictions = [], {}
    conditions = ["clean", "grayscale", "hue_rotation", "patch_shuffle"]
    for model_name in all_models:
        feature_model = "clip_vit_b32" if model_name == "clip_zeroshot" else model_name
        clean_probabilities = probability_fn(model_name, feature_cache[feature_model]["clean"])
        clean_predictions = clean_probabilities.argmax(1)
        for condition in conditions:
            probabilities = probability_fn(model_name, feature_cache[feature_model][condition])
            row, condition_predictions = metric_row(
                model_name, condition, probabilities, labels,
                None if condition == "clean" else clean_predictions,
            )
            rows.append(row)
            predictions[(model_name, condition)] = condition_predictions
    results = pd.DataFrame(rows)
    clean_accuracy = results.query("condition == 'clean'").set_index("model").accuracy
    results["accuracy_delta_vs_clean"] = results.apply(lambda row: row.accuracy - clean_accuracy[row.model], axis=1)
    results.to_csv(results_dir / "clean_color_patch_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=results, x="condition", y="accuracy", hue="model", ax=axes[0])
    axes[0].set_title("Absolute accuracy")
    sns.barplot(data=results.query("condition != 'clean'"), x="condition", y="prediction_consistency", hue="model", ax=axes[1])
    axes[1].set_title("Prediction consistency with clean")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(results_dir / "clean_color_patch_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return results, predictions


def evaluate_cue_conflicts(all_models, accepted, conflict_features, probability_fn, class_names, results_dir):
    content_labels = torch.tensor(accepted.content_label.to_numpy())
    style_labels = torch.tensor(accepted.style_label.to_numpy())
    rows, example_rows = [], []
    for model_name in all_models:
        feature_model = "clip_vit_b32" if model_name == "clip_zeroshot" else model_name
        probabilities = probability_fn(model_name, conflict_features[feature_model])
        predictions = probabilities.argmax(1)
        shape_mask, texture_mask = predictions.eq(content_labels), predictions.eq(style_labels)
        n_shape, n_texture = int(shape_mask.sum()), int(texture_mask.sum())
        n_other, total = len(predictions) - n_shape - n_texture, len(predictions)
        denominator = n_shape + n_texture
        rows.append({
            "model": model_name, "n_shape": n_shape, "n_texture": n_texture, "n_other": n_other, "n_total": total,
            "shape_bias_percent": 100 * n_shape / denominator if denominator else np.nan,
            "coverage_percent": 100 * denominator / total,
        })
        for index, prediction in enumerate(predictions.tolist()):
            example_rows.append({
                "conflict_id": accepted.iloc[index].conflict_id, "model": model_name,
                "prediction_index": prediction, "prediction_class": class_names[prediction],
                "confidence": probabilities[index, prediction].item(),
            })
    results, per_example = pd.DataFrame(rows), pd.DataFrame(example_rows)
    results.to_csv(results_dir / "cue_conflict_shape_bias.csv", index=False)
    per_example.to_csv(results_dir / "cue_conflict_predictions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=results, x="model", y="shape_bias_percent", ax=axes[0])
    sns.barplot(data=results, x="model", y="coverage_percent", ax=axes[1])
    axes[0].set_title("Shape bias among intended-label decisions")
    axes[1].set_title("Shape/texture decision coverage")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(results_dir / "cue_conflict_shape_bias_and_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return results, per_example


def evaluate_translation(all_models, feature_cache, labels, probability_fn, deltas, directions, results_dir):
    rows = []
    for model_name in all_models:
        feature_model = "clip_vit_b32" if model_name == "clip_zeroshot" else model_name
        clean_predictions = probability_fn(model_name, feature_cache[feature_model]["clean"]).argmax(1)
        rows.append({"model": model_name, "delta": 0, "direction": "identity",
                     "accuracy": accuracy_score(labels, clean_predictions), "consistency": 1.0})
        for delta in deltas[1:]:
            for direction in directions:
                predictions = probability_fn(model_name, feature_cache[feature_model][f"translation_{delta}_{direction}"]).argmax(1)
                rows.append({"model": model_name, "delta": delta, "direction": direction,
                             "accuracy": accuracy_score(labels, predictions),
                             "consistency": predictions.eq(clean_predictions).float().mean().item()})
    by_direction = pd.DataFrame(rows)
    summary = by_direction.groupby(["model", "delta"], as_index=False)[["accuracy", "consistency"]].mean()
    by_direction.to_csv(results_dir / "translation_metrics_by_direction.csv", index=False)
    summary.to_csv(results_dir / "translation_metrics_mean.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    sns.lineplot(data=summary, x="delta", y="accuracy", hue="model", marker="o", ax=axes[0])
    sns.lineplot(data=summary, x="delta", y="consistency", hue="model", marker="o", ax=axes[1])
    axes[0].set_title("Accuracy vs displacement")
    axes[1].set_title("Prediction consistency vs displacement")
    for axis in axes:
        axis.set_xticks(deltas)
        axis.set_xlabel("Displacement (pixels)")
    fig.tight_layout()
    fig.savefig(results_dir / "translation_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return by_direction, summary


def save_informative_examples(predictions, accepted, all_models, results_dir):
    wide = predictions.pivot(index="conflict_id", columns="model", values="prediction_class")
    wide["n_unique_predictions"] = wide[all_models].nunique(axis=1)
    wide = wide.join(accepted.set_index("conflict_id")[["path", "content_class", "style_class"]])
    examples = pd.concat([wide.sort_values("n_unique_predictions", ascending=False).head(8), wide.query("n_unique_predictions == 1").head(4)]).drop_duplicates().head(12)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for axis, (_, row) in zip(axes.flat, examples.iterrows()):
        axis.imshow(Image.open(row.path))
        axis.axis("off")
        prediction_text = "\n".join(f"{model}: {row[model]}" for model in all_models)
        axis.set_title(f"shape={row.content_class}; texture={row.style_class}\n{prediction_text}", fontsize=8)
    fig.suptitle("Cue-conflict agreements, disagreements, and failures", fontsize=15)
    fig.tight_layout()
    fig.savefig(results_dir / "cue_conflict_informative_examples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    examples.reset_index().to_csv(results_dir / "cue_conflict_informative_examples.csv", index=False)
    return examples
