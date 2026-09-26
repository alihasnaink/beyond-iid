from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchvision.datasets import CIFAR100

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from data.cifar10 import CIFAR10_CLASSES  # noqa: E402
from evaluation.failure_analysis import plot_failures, select_failures  # noqa: E402
from evaluation.metrics import evaluate_unknownness, known_accuracy  # noqa: E402
from scores import (DiagonalMahalanobis, calibrate_placeholder, energy_unknownness,  # noqa: E402
                    mls_unknownness, msp_unknownness, placeholder_unknownness)


def _load_outputs(task_dir, method):
    cache = Path(task_dir) / "cache"
    return (torch.load(cache / f"{method}_known_outputs.pt", weights_only=False),
            torch.load(cache / f"{method}_unknown_outputs.pt", weights_only=False))


def _score_bundle(score_fn, known, unknown):
    return {
        "validation": score_fn(known["validation"]), "test": score_fn(known["test"]),
        "near": score_fn(unknown["near"]), "far": score_fn(unknown["far"]),
    }


def _row(method, score_name, bundle, known, threshold=None):
    values = evaluate_unknownness(bundle["validation"], bundle["test"], bundle["near"], bundle["far"], threshold)
    return {"method": method, "score": score_name,
            "closed_set_accuracy": known_accuracy(known["test"]["logits"], known["test"]["labels"]), **values}


def _plot_score_distributions(score_bundles, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for axis, (name, bundle) in zip(axes, score_bundles.items()):
        for split, color in [("test", "#2672B8"), ("near", "#E68613"), ("far", "#B52B42")]:
            values = np.asarray(bundle[split])
            sns.histplot(values, bins=55, stat="density", element="step", fill=False,
                         label={"test": "CIFAR-10 known", "near": "near unknown", "far": "far unknown"}[split],
                         color=color, ax=axis)
        threshold = float(np.quantile(np.asarray(bundle["validation"]), .95))
        axis.axvline(threshold, color="black", linestyle="--", label="validation $q_{.95}$")
        axis.set_title(f"Vanilla {name}"); axis.set_xlabel("unknownness (larger = more unknown)")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(results_dir / "vanilla_score_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _absorption_table(method_outputs, method_scores, thresholds):
    rows = []
    for method, unknown in method_outputs.items():
        for group in ["near", "far"]:
            output, scores = unknown[group], np.asarray(method_scores[method][group])
            predictions = output["logits"].argmax(1).numpy()
            for class_name in sorted(set(output["class_names"])):
                mask = np.asarray(output["class_names"]) == class_name
                accepted = mask & (scores <= thresholds[method])
                if accepted.any():
                    counts = np.bincount(predictions[accepted], minlength=10)
                    dominant, share = CIFAR10_CLASSES[int(counts.argmax())], float(counts.max() / accepted.sum())
                else:
                    dominant, share = "none", np.nan
                rows.append({"method": method, "group": group, "unknown_class": class_name,
                             "acceptance_rate": float(accepted.sum() / mask.sum()),
                             "dominant_known_prediction": dominant, "dominant_prediction_share": share})
    return pd.DataFrame(rows)


def _score_disagreements(score_frame, vanilla_known, vanilla_unknown, pairs_per_comparison=5):
    metadata = pd.DataFrame({
        "split": (["known_test"] * len(vanilla_known["test"]["labels"]) +
                  ["near"] * len(vanilla_unknown["near"]["labels"]) +
                  ["far"] * len(vanilla_unknown["far"]["labels"])),
        "identifier": (vanilla_known["test"]["identifiers"] + vanilla_unknown["near"]["identifiers"] +
                       vanilla_unknown["far"]["identifiers"]),
        "class_name": (vanilla_known["test"]["class_names"] + vanilla_unknown["near"]["class_names"] +
                       vanilla_unknown["far"]["class_names"]),
    })
    logits = torch.cat([vanilla_known["test"]["logits"], vanilla_unknown["near"]["logits"],
                        vanilla_unknown["far"]["logits"]])
    metadata["predicted_known_class"] = [CIFAR10_CLASSES[i] for i in logits.argmax(1).tolist()]
    ranks = score_frame.rank(pct=True)
    rows = []
    names = list(score_frame.columns)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1:]:
            gaps = (ranks[first] - ranks[second]).abs()
            for index in gaps.nlargest(pairs_per_comparison).index:
                rows.append({**metadata.iloc[index].to_dict(), "score_a": first, "score_b": second,
                             "value_a": float(score_frame.loc[index, first]),
                             "value_b": float(score_frame.loc[index, second]),
                             "percentile_a": float(ranks.loc[index, first]),
                             "percentile_b": float(ranks.loc[index, second]),
                             "absolute_rank_gap": float(gaps.loc[index])})
    return pd.DataFrame(rows).sort_values("absolute_rank_gap", ascending=False)


def _write_report_notes(posthoc, trained, absorption, correlations, failures, path):
    best_near = posthoc.loc[posthoc.auroc_near.idxmax()]
    best_far = posthoc.loc[posthoc.auroc_far.idxmax()]
    best_all = posthoc.loc[posthoc.auroc_all.idxmax()]
    vanilla = trained.query("method == 'Vanilla' and score == 'MLS'").iloc[0]
    gcsc = trained.query("method == 'GCSC' and score == 'MLS'").iloc[0]
    proser = trained.query("method == 'PROSER' and score == 'MLS'").iloc[0]
    placeholder = trained.query("method == 'PROSER' and score == 'placeholder'").iloc[0]
    hardest = (absorption.query("method == 'Vanilla'").sort_values("acceptance_rate", ascending=False)
               .groupby("group", sort=False).head(2))
    corr_pairs = correlations.where(np.triu(np.ones(correlations.shape), 1).astype(bool)).stack()
    strongest_pair, weakest_pair = corr_pairs.idxmax(), corr_pairs.idxmin()
    lines = [
        "# Task 4 report notes (generated from this run)", "",
        "These are quantitative prompts, not claims to copy without inspecting the figures and failure images.", "",
        "## Frozen-score comparison", "",
        f"- Best near AUROC: **{best_near.score}** ({best_near.auroc_near:.4f}); best far AUROC: "
        f"**{best_far.score}** ({best_far.auroc_far:.4f}); best all-unknown AUROC: **{best_all.score}** "
        f"({best_all.auroc_all:.4f}).",
        f"- Strongest score rank agreement is {strongest_pair[0]} vs {strongest_pair[1]} "
        f"(Spearman {corr_pairs.max():.3f}); weakest is {weakest_pair[0]} vs {weakest_pair[1]} "
        f"({corr_pairs.min():.3f}). Inspect `vanilla_score_rank_disagreements.csv` before explaining why.", "",
        "## Positive augmentation and placeholders", "",
        f"- GCSC minus Vanilla: CSA {gcsc.closed_set_accuracy - vanilla.closed_set_accuracy:+.4f}, "
        f"near AUROC {gcsc.auroc_near - vanilla.auroc_near:+.4f}, far AUROC "
        f"{gcsc.auroc_far - vanilla.auroc_far:+.4f}.",
        f"- PROSER-MLS minus Vanilla: CSA {proser.closed_set_accuracy - vanilla.closed_set_accuracy:+.4f}, "
        f"near AUROC {proser.auroc_near - vanilla.auroc_near:+.4f}, far AUROC "
        f"{proser.auroc_far - vanilla.auroc_far:+.4f}.",
        f"- PROSER placeholder versus its MLS: near AUROC {placeholder.auroc_near - proser.auroc_near:+.4f}, "
        f"far AUROC {placeholder.auroc_far - proser.auroc_far:+.4f}. Relate any gains to classifier/data "
        "placeholders, while noting that between-class interpolations cannot cover every unknown direction.", "",
        "## Semantic difficulty and failures", "",
    ]
    for row in hardest.itertuples(index=False):
        lines.append(f"- Vanilla most often accepts {row.group} class **{row.unknown_class}** "
                     f"({row.acceptance_rate:.1%}); accepted examples are most often labeled "
                     f"**{row.dominant_known_prediction}**.")
    lines += ["", f"- The qualitative sheet contains {len(failures)} accepted unknowns. Discuss whether each "
              "mapping is semantically plausible or surprising; do not use these observations to tune the system.", "",
              "## Interpretation reminder", "",
              "AUROC summarizes ranking across thresholds; validation-calibrated rejection describes one operating "
              "point. Discuss both, and do not infer broad unknown robustness from the fixed near/far class sets."]
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def evaluate(task_dir=TASK_DIR, download=True):
    task_dir = Path(task_dir).resolve(); results = task_dir / "results"; results.mkdir(parents=True, exist_ok=True)
    for method in ["vanilla", "gcsc", "proser"]:
        for kind in ["known", "unknown"]:
            path = task_dir / "cache" / f"{method}_{kind}_outputs.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}. Run extract_outputs.py after training.")

    vanilla_known, vanilla_unknown = _load_outputs(task_dir, "vanilla")
    gcsc_known, gcsc_unknown = _load_outputs(task_dir, "gcsc")
    proser_known, proser_unknown = _load_outputs(task_dir, "proser")

    mahalanobis = DiagonalMahalanobis(epsilon=1e-6).fit(
        vanilla_known["train"]["features"], vanilla_known["train"]["labels"], 10)
    torch.save(mahalanobis.state_dict(), task_dir / "cache/vanilla_mahalanobis.pt")
    vanilla_scores = {
        "MSP": _score_bundle(lambda x: msp_unknownness(x["logits"]), vanilla_known, vanilla_unknown),
        "MLS": _score_bundle(lambda x: mls_unknownness(x["logits"]), vanilla_known, vanilla_unknown),
        "Energy": _score_bundle(lambda x: energy_unknownness(x["logits"]), vanilla_known, vanilla_unknown),
        "Mahalanobis": _score_bundle(lambda x: mahalanobis.score(x["features"]), vanilla_known, vanilla_unknown),
    }
    posthoc = pd.DataFrame([_row("Vanilla", name, bundle, vanilla_known)
                            for name, bundle in vanilla_scores.items()])
    posthoc.to_csv(results / "posthoc_score_comparison.csv", index=False)

    gcsc_mls = _score_bundle(lambda x: mls_unknownness(x["logits"]), gcsc_known, gcsc_unknown)
    proser_mls = _score_bundle(lambda x: mls_unknownness(x["logits"]), proser_known, proser_unknown)
    proser_raw = _score_bundle(lambda x: placeholder_unknownness(x["logits"], x["dummy_logits"]),
                               proser_known, proser_unknown)
    placeholder_bias = calibrate_placeholder(proser_raw["validation"], .95)
    proser_placeholder = {key: value + placeholder_bias for key, value in proser_raw.items()}

    trained = pd.DataFrame([
        _row("Vanilla", "MLS", vanilla_scores["MLS"], vanilla_known),
        _row("GCSC", "MLS", gcsc_mls, gcsc_known),
        _row("PROSER", "MLS", proser_mls, proser_known),
        {**_row("PROSER", "placeholder", proser_placeholder, proser_known, threshold=0.0),
         "placeholder_bias": placeholder_bias},
    ])
    trained.to_csv(results / "trained_method_comparison.csv", index=False)

    _plot_score_distributions({k: vanilla_scores[k] for k in ["MSP", "MLS", "Mahalanobis"]}, results)
    score_frame = pd.DataFrame({name: np.r_[np.asarray(bundle["test"]), np.asarray(bundle["near"]),
                                                    np.asarray(bundle["far"])]
                                for name, bundle in vanilla_scores.items()})
    correlations = score_frame.corr(method="spearman")
    correlations.to_csv(results / "vanilla_score_spearman_correlations.csv")
    disagreements = _score_disagreements(score_frame, vanilla_known, vanilla_unknown)
    disagreements.to_csv(results / "vanilla_score_rank_disagreements.csv", index=False)
    fig, axis = plt.subplots(figsize=(6, 5)); sns.heatmap(correlations, annot=True, vmin=-1, vmax=1, cmap="vlag", ax=axis)
    axis.set_title("Rank agreement among Vanilla unknownness scores")
    fig.tight_layout(); fig.savefig(results / "vanilla_score_agreement.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    method_outputs = {"Vanilla": vanilla_unknown, "GCSC": gcsc_unknown, "PROSER": proser_unknown}
    method_scores = {"Vanilla": vanilla_scores["MLS"], "GCSC": gcsc_mls, "PROSER": proser_mls}
    thresholds = {name: float(np.quantile(np.asarray(bundle["validation"]), .95)) for name, bundle in method_scores.items()}
    absorption = _absorption_table(method_outputs, method_scores, thresholds)
    absorption.to_csv(results / "unknown_class_absorption.csv", index=False)

    vanilla_threshold = thresholds["Vanilla"]
    failures = select_failures(vanilla_unknown,
                               {g: vanilla_scores["MLS"][g].numpy() for g in ["near", "far"]},
                               vanilla_threshold, count=3)
    failures.to_csv(results / "vanilla_mls_failure_cases.csv", index=False)
    cifar100_test = CIFAR100(task_dir / "data/raw", train=False, download=download)
    plot_failures(failures, cifar100_test, results / "vanilla_mls_failure_cases.png")
    report_notes = _write_report_notes(posthoc, trained, absorption, correlations, failures,
                                       results / "report_notes.md")

    audit = {
        "known_dataset": "CIFAR-10", "unknown_dataset": "fixed CIFAR-100 test classes",
        "cifar100_train_used": False, "seed": 6304, "validation_quantile": 0.95,
        "larger_score_means_unknown": True, "mahalanobis_epsilon": 1e-6,
        "proser_placeholder_bias": placeholder_bias,
        "artifacts": sorted({path.name for path in results.iterdir() if path.is_file()} | {"completion_audit.json"}),
    }
    (results / "completion_audit.json").write_text(json.dumps(audit, indent=2))
    return {"posthoc": posthoc, "trained": trained, "absorption": absorption,
            "failures": failures, "correlations": correlations, "disagreements": disagreements,
            "report_notes": report_notes, "audit": audit}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 4 open-set recognition")
    parser.parse_args()
    output = evaluate()
    print("\nFrozen Vanilla score comparison\n", output["posthoc"].to_string(index=False))
    print("\nTraining-method comparison\n", output["trained"].to_string(index=False))


if __name__ == "__main__":
    main()
