from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F


def rowwise_cosine(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=1)


def compute_stability(model_names, feature_cache, conflict_features, accepted, directions, results_dir):
    rows = []
    content_positions = torch.tensor(accepted.content_eval_position.to_numpy(), dtype=torch.long)
    for model_name in model_names:
        clean = feature_cache[model_name]["clean"]
        for condition in ["grayscale", "patch_shuffle"]:
            values = rowwise_cosine(clean, feature_cache[model_name][condition])
            rows.append({"backbone": model_name, "intervention": condition,
                         "mean_cosine_stability": values.mean().item(), "std": values.std().item(), "n_pairs": len(values)})
        translated = torch.cat([rowwise_cosine(clean, feature_cache[model_name][f"translation_32_{direction}"]) for direction in directions])
        rows.append({"backbone": model_name, "intervention": "translation_32px_four_direction_mean",
                     "mean_cosine_stability": translated.mean().item(), "std": translated.std().item(), "n_pairs": len(translated)})
        cue = rowwise_cosine(clean[content_positions], conflict_features[model_name])
        rows.append({"backbone": model_name, "intervention": "cue_conflict",
                     "mean_cosine_stability": cue.mean().item(), "std": cue.std().item(), "n_pairs": len(cue)})
    results = pd.DataFrame(rows)
    results.to_csv(results_dir / "representation_cosine_stability.csv", index=False)
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results, x="intervention", y="mean_cosine_stability", hue="backbone", ax=axis)
    axis.tick_params(axis="x", rotation=15)
    axis.set_ylim(0, 1)
    axis.set_title("Backbone representation stability")
    fig.tight_layout()
    fig.savefig(results_dir / "representation_cosine_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return results


def compare_prediction_representation(all_models, accepted, standard_results, standard_predictions, translation_summary,
                                      stability_results, conflict_features, probability_fn, results_dir):
    lookup = stability_results.set_index(["backbone", "intervention"]).mean_cosine_stability
    rows = []
    content_positions = torch.tensor(accepted.content_eval_position.to_numpy(), dtype=torch.long)
    for model_name in all_models:
        backbone = "clip_vit_b32" if model_name == "clip_zeroshot" else model_name
        for intervention in ["grayscale", "patch_shuffle"]:
            consistency = standard_results.query("model == @model_name and condition == @intervention").prediction_consistency.iloc[0]
            rows.append({"model": model_name, "intervention": intervention, "prediction_consistency": consistency,
                         "mean_cosine_stability": lookup[backbone, intervention]})
        consistency = translation_summary.query("model == @model_name and delta == 32").consistency.iloc[0]
        rows.append({"model": model_name, "intervention": "translation_32px", "prediction_consistency": consistency,
                     "mean_cosine_stability": lookup[backbone, "translation_32px_four_direction_mean"]})
        clean_prediction = standard_predictions[(model_name, "clean")][content_positions]
        cue_consistency = probability_fn(model_name, conflict_features[backbone]).argmax(1).eq(clean_prediction).float().mean().item()
        rows.append({"model": model_name, "intervention": "cue_conflict", "prediction_consistency": cue_consistency,
                     "mean_cosine_stability": lookup[backbone, "cue_conflict"]})
    results = pd.DataFrame(rows)
    results.to_csv(results_dir / "prediction_representation_alignment.csv", index=False)
    fig, axis = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=results, x="mean_cosine_stability", y="prediction_consistency", hue="model", style="intervention", s=110, ax=axis)
    for row in results.itertuples():
        axis.annotate(row.intervention.replace("_", " "), (row.mean_cosine_stability, row.prediction_consistency), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axis.set_title("Prediction stability vs representation stability")
    fig.tight_layout()
    fig.savefig(results_dir / "prediction_representation_alignment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return results
