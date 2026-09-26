from __future__ import annotations

import inspect
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def _fit_tsne(features, cfg):
    kwargs = dict(n_components=2, perplexity=cfg.tsne_perplexity, init="pca", learning_rate="auto", random_state=cfg.seed)
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = cfg.tsne_iterations
    else:
        kwargs["n_iter"] = cfg.tsne_iterations
    return TSNE(**kwargs).fit_transform(features.numpy())


def make_tsne_plots(model_names, feature_cache, conflict_features, accepted, eval_labels, class_names, cfg, results_dir):
    rng = np.random.default_rng(cfg.seed)
    per_class = cfg.tsne_pairs // len(class_names)
    eval_positions = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(eval_labels.numpy() == class_idx), per_class, replace=False)
        for class_idx in range(len(class_names))
    ]))
    per_cell = cfg.tsne_pairs // 10
    cue_positions = np.sort(np.concatenate([
        rng.choice(group.index.to_numpy(), per_cell, replace=False)
        for _, group in accepted.reset_index(drop=True).groupby(["pair_id", "content_class", "style_class"])
    ]))
    settings = {
        "method": "t-SNE", "perplexity": cfg.tsne_perplexity, "init": "pca",
        "learning_rate": "auto", "iterations": cfg.tsne_iterations, "seed": cfg.seed,
        "paired_eval_positions": eval_positions.tolist(), "paired_cue_positions": cue_positions.tolist(),
    }
    (results_dir / "tsne_settings_and_subset.json").write_text(json.dumps(settings, indent=2))
    content_labels = torch.tensor(accepted.content_label.to_numpy())
    for model_name in model_names:
        clean = feature_cache[model_name]["clean"]
        content_positions = torch.tensor(accepted.iloc[cue_positions].content_eval_position.to_numpy(), dtype=torch.long)
        specs = [
            ("grayscale", clean[eval_positions], feature_cache[model_name]["grayscale"][eval_positions], eval_labels[eval_positions]),
            ("patch shuffle", clean[eval_positions], feature_cache[model_name]["patch_shuffle"][eval_positions], eval_labels[eval_positions]),
            ("translation right 32px", clean[eval_positions], feature_cache[model_name]["translation_32_right"][eval_positions], eval_labels[eval_positions]),
            ("cue conflict", clean[content_positions], conflict_features[model_name][cue_positions], content_labels[cue_positions]),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(16, 13))
        for axis, (condition, clean_features, transformed_features, labels) in zip(axes.flat, specs):
            embedding = _fit_tsne(torch.cat([clean_features, transformed_features]).float(), cfg)
            n_items, labels_array = len(clean_features), labels.numpy()
            for class_idx, class_name in enumerate(class_names):
                mask = labels_array == class_idx
                color = plt.cm.tab10(class_idx)
                axis.scatter(embedding[:n_items][mask, 0], embedding[:n_items][mask, 1], s=16, alpha=.65, marker="o", color=color, label=class_name)
                axis.scatter(embedding[n_items:][mask, 0], embedding[n_items:][mask, 1], s=22, alpha=.70, marker="x", color=color)
            axis.set_title(f"{condition}: circle=clean, x=transformed")
            axis.set_xticks([])
            axis.set_yticks([])
        handles, labels_legend = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc="center right", title="Ground-truth/content class")
        fig.suptitle(f"Paired t-SNE projections - {model_name}\n(each panel fitted to combined clean + transformed features)", fontsize=15)
        fig.tight_layout(rect=[0, 0, .91, .95])
        fig.savefig(results_dir / f"tsne_clean_vs_transformed_{model_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    return settings
