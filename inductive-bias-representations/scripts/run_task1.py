from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.datasets import STL10

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from analysis.evaluate_bias import (  # noqa: E402
    evaluate_cue_conflicts,
    evaluate_standard,
    evaluate_translation,
    save_informative_examples,
)
from analysis.feature_similarity import compare_prediction_representation, compute_stability  # noqa: E402
from analysis.representation import make_tsne_plots  # noqa: E402
from configs.task1_config import (  # noqa: E402
    CLASS_NAMES,
    CUE_PAIRS,
    TRANSLATION_DELTAS,
    TRANSLATION_DIRECTIONS,
    Config,
)
from data.make_cue_conflicts import (  # noqa: E402
    ConflictDataset,
    finalize_candidates,
    generate_candidates,
    save_qc_sheets,
)
from data.make_subset import make_splits  # noqa: E402
from data.transforms import IndexedSTL  # noqa: E402
from models.backbones import MODEL_BUILDERS, build_backbone, extract_features  # noqa: E402
from models.linear_probe import predict_head, seed_everything, train_linear_head  # noqa: E402


def _atomic_torch_save(value, path: Path):
    """Write a cache without leaving a truncated target after interruption."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


class Task1Pipeline:
    def __init__(self, cfg: Config | None = None, task_dir: Path | None = None):
        self.cfg = cfg or Config()
        self.task_dir = Path(task_dir or TASK_DIR).resolve()
        self.data_dir = self.task_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.cache_dir = self.data_dir / "cache"
        self.conflict_dir = self.data_dir / "generated" / "cue_conflicts"
        self.results_dir = self.task_dir / "results"
        for path in [self.raw_dir, self.cache_dir, self.conflict_dir, self.results_dir]:
            path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_tag = hashlib.sha1(json.dumps(asdict(self.cfg), sort_keys=True).encode()).hexdigest()[:10]
        self.all_models = ["resnet50", "vit_b16", "clip_vit_b32", "clip_zeroshot"]
        self.heads = {}
        self._save_metadata()

    def _save_json(self, obj, path):
        path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))

    def _save_metadata(self):
        import torchvision

        try:
            import open_clip
            open_clip_version = getattr(open_clip, "__version__", "unknown")
        except ImportError:
            open_clip_version = "not-installed"
        self._save_json({"config": asdict(self.cfg), "classes": CLASS_NAMES, "cue_pairs": CUE_PAIRS}, self.results_dir / "experiment_config.json")
        self._save_json({
            "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "open_clip": open_clip_version, "device": str(self.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }, self.results_dir / "environment_versions.json")

    def prepare_data(self, download=True):
        seed_everything(self.cfg.seed)
        self.train_dataset = STL10(root=self.raw_dir, split="train", download=download)
        self.test_dataset = STL10(root=self.raw_dir, split="test", download=download)
        self.train_idx, self.val_idx, self.selected_test_idx, self.subset_manifest = make_splits(
            self.train_dataset, self.test_dataset, CLASS_NAMES, self.cfg.seed,
            self.cfg.test_per_class, self.results_dir,
        )
        counts = self.subset_manifest.class_name.value_counts().reindex(CLASS_NAMES)
        if len(self.subset_manifest) != 500 or counts.nunique() != 1:
            print("WARNING: evaluation subset is not 50 examples per class; document this imbalance.")
        return self.subset_manifest

    def train_heads(self):
        if not hasattr(self, "train_dataset"):
            self.prepare_data()
        histories = []
        train_data = IndexedSTL(self.train_dataset, self.train_idx, self.cfg, "clean")
        val_data = IndexedSTL(self.train_dataset, self.val_idx, self.cfg, "clean")
        for model_name, builder in MODEL_BUILDERS.items():
            feature_path = self.cache_dir / f"trainval_{model_name}_{self.run_tag}.pt"
            if feature_path.exists():
                cached = torch.load(feature_path, map_location="cpu", weights_only=False)
            else:
                backbone = build_backbone(model_name, self.device)
                train_x, train_y, _ = extract_features(backbone, train_data, self.device, self.cfg.batch_size, self.cfg.num_workers, f"{model_name} train")
                val_x, val_y, _ = extract_features(backbone, val_data, self.device, self.cfg.batch_size, self.cfg.num_workers, f"{model_name} validation")
                cached = {"train_x": train_x, "train_y": train_y, "val_x": val_x, "val_y": val_y}
                torch.save(cached, feature_path)
                del backbone
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            head, history = train_linear_head(**cached, feature_dim=builder.dim, model_name=model_name,
                                              cfg=self.cfg, device=self.device, n_classes=len(CLASS_NAMES))
            self.heads[model_name] = head
            histories.append(history)
        history = pd.concat(histories, ignore_index=True)
        history.to_csv(self.results_dir / "linear_head_training_history.csv", index=False)
        torch.save({name: head.state_dict() for name, head in self.heads.items()}, self.cache_dir / f"linear_heads_{self.run_tag}.pt")
        return history

    def generate_cue_conflicts(self, manual_rejections=None):
        if not hasattr(self, "test_dataset"):
            self.prepare_data()
        candidates = generate_candidates(
            self.test_dataset, self.subset_manifest, CLASS_NAMES, CUE_PAIRS, self.cfg,
            self.cache_dir, self.conflict_dir, self.device, manual_rejections,
        )
        accepted, counts, quota = finalize_candidates(candidates, self.subset_manifest, self.cfg)
        candidates.to_csv(self.results_dir / "cue_conflict_candidates_qc.csv", index=False)
        accepted.to_csv(self.results_dir / "cue_conflict_accepted_manifest.csv", index=False)
        counts.to_csv(self.results_dir / "cue_conflict_qc_counts.csv", index=False)
        sheets = save_qc_sheets(candidates, self.results_dir)
        self.candidates, self.accepted, self.cue_quota = candidates, accepted, quota
        return candidates, accepted, counts, sheets

    def load_accepted_conflicts(self):
        self.accepted = pd.read_csv(self.results_dir / "cue_conflict_accepted_manifest.csv")
        self.cue_quota = int(self.accepted.groupby(["pair_id", "content_class", "style_class"]).size().min())
        return self.accepted

    def extract_evaluation_features(self):
        if not hasattr(self, "test_dataset"):
            self.prepare_data()
        if not hasattr(self, "accepted"):
            self.load_accepted_conflicts()
        feature_cache, conflict_cache = {}, {}
        base_conditions = {"clean": "clean", "grayscale": "grayscale", "hue_rotation": "hue", "patch_shuffle": "patch"}
        expected = set(base_conditions) | {
            f"translation_{delta}_{direction}"
            for delta in TRANSLATION_DELTAS[1:] for direction in TRANSLATION_DIRECTIONS
        }
        cue_tag = hashlib.sha1("|".join(self.accepted.conflict_id).encode()).hexdigest()[:10]
        for model_name in MODEL_BUILDERS:
            cache_path = self.cache_dir / f"eval_features_{model_name}_{self.run_tag}.pt"
            cached = torch.load(cache_path, map_location="cpu", weights_only=False) if cache_path.exists() else {}
            if not isinstance(cached, dict):
                cached = {}
            # Ignore stale/unknown entries while retaining completed conditions.
            cached = {key: value for key, value in cached.items() if key in expected}
            cue_path = self.cache_dir / f"cue_features_{model_name}_{self.run_tag}_{cue_tag}.pt"
            cue_cached = torch.load(cue_path, map_location="cpu", weights_only=False) if cue_path.exists() else None
            if set(cached) != expected or cue_cached is None or len(cue_cached) != len(self.accepted):
                backbone = build_backbone(model_name, self.device)
                try:
                    for condition, mode in base_conditions.items():
                        if condition not in cached:
                            dataset = IndexedSTL(self.test_dataset, self.selected_test_idx, self.cfg, mode)
                            cached[condition] = extract_features(backbone, dataset, self.device, self.cfg.batch_size, self.cfg.num_workers, f"{model_name} {condition}")[0]
                            _atomic_torch_save(cached, cache_path)
                    for delta in TRANSLATION_DELTAS[1:]:
                        for direction in TRANSLATION_DIRECTIONS:
                            key = f"translation_{delta}_{direction}"
                            if key not in cached:
                                dataset = IndexedSTL(self.test_dataset, self.selected_test_idx, self.cfg, "translation", delta, direction)
                                cached[key] = extract_features(backbone, dataset, self.device, self.cfg.batch_size, self.cfg.num_workers, f"{model_name} {key}")[0]
                                _atomic_torch_save(cached, cache_path)
                    if cue_cached is None or len(cue_cached) != len(self.accepted):
                        dataset = ConflictDataset(self.accepted, self.cfg.image_size)
                        cue_cached = extract_features(backbone, dataset, self.device, self.cfg.batch_size, self.cfg.num_workers, f"{model_name} cue conflicts")[0]
                        _atomic_torch_save(cue_cached, cue_path)
                finally:
                    del backbone
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            feature_cache[model_name], conflict_cache[model_name] = cached, cue_cached
        self.feature_cache, self.conflict_features = feature_cache, conflict_cache
        self.eval_labels = torch.tensor(self.subset_manifest.class_index.to_numpy(), dtype=torch.long)
        return feature_cache, conflict_cache

    def _prepare_probability_functions(self):
        if not self.heads:
            self.train_heads()
        import open_clip

        clip_model = build_backbone("clip_vit_b32", self.device)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        with torch.inference_mode():
            tokens = tokenizer([f"a photo of a {name}." for name in CLASS_NAMES]).to(self.device)
            self.text_features = clip_model.model.encode_text(tokens, normalize=True).float().cpu()
            self.clip_logit_scale = clip_model.model.logit_scale.exp().float().cpu().clamp(max=100)
        del clip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def probability_fn(self, model_name, features):
        if model_name == "clip_zeroshot":
            return (self.clip_logit_scale * features.float() @ self.text_features.T).softmax(1)
        return predict_head(self.heads[model_name], features)

    def evaluate(self):
        if not hasattr(self, "feature_cache"):
            self.extract_evaluation_features()
        self._prepare_probability_functions()
        standard, predictions = evaluate_standard(self.all_models, self.feature_cache, self.eval_labels, self.probability_fn, self.results_dir)
        cue, cue_predictions = evaluate_cue_conflicts(self.all_models, self.accepted, self.conflict_features, self.probability_fn, CLASS_NAMES, self.results_dir)
        by_direction, translation = evaluate_translation(
            self.all_models, self.feature_cache, self.eval_labels, self.probability_fn,
            TRANSLATION_DELTAS, TRANSLATION_DIRECTIONS, self.results_dir,
        )
        stability = compute_stability(MODEL_BUILDERS, self.feature_cache, self.conflict_features, self.accepted, TRANSLATION_DIRECTIONS, self.results_dir)
        alignment = compare_prediction_representation(
            self.all_models, self.accepted, standard, predictions, translation, stability,
            self.conflict_features, self.probability_fn, self.results_dir,
        )
        tsne_settings = make_tsne_plots(
            MODEL_BUILDERS, self.feature_cache, self.conflict_features, self.accepted,
            self.eval_labels, CLASS_NAMES, self.cfg, self.results_dir,
        )
        examples = save_informative_examples(cue_predictions, self.accepted, self.all_models, self.results_dir)
        self._completion_audit()
        return {
            "standard": standard, "cue_conflicts": cue, "translation": translation,
            "stability": stability, "alignment": alignment, "examples": examples,
            "tsne_settings": tsne_settings, "translation_by_direction": by_direction,
        }

    def _completion_audit(self):
        required = [
            "experiment_config.json", "environment_versions.json", "train_validation_split.json",
            "selected_test_identifiers.csv", "linear_head_training_history.csv",
            "clean_color_patch_metrics.csv", "clean_color_patch_comparison.png",
            "cue_conflict_candidates_qc.csv", "cue_conflict_accepted_manifest.csv", "cue_conflict_qc_counts.csv",
            "cue_conflict_shape_bias.csv", "cue_conflict_shape_bias_and_coverage.png", "cue_conflict_predictions.csv",
            "translation_metrics_by_direction.csv", "translation_metrics_mean.csv", "translation_curves.png",
            "representation_cosine_stability.csv", "representation_cosine_stability.png",
            "prediction_representation_alignment.csv", "prediction_representation_alignment.png",
            "tsne_settings_and_subset.json", "cue_conflict_informative_examples.png", "cue_conflict_informative_examples.csv",
        ]
        required += [f"cue_conflict_qc_contact_sheet_{i:02d}.png" for i in range(1, math.ceil(len(self.candidates if hasattr(self, 'candidates') else pd.read_csv(self.results_dir / 'cue_conflict_candidates_qc.csv')) / 50) + 1)]
        required += [f"tsne_clean_vs_transformed_{name}.png" for name in MODEL_BUILDERS]
        missing = [name for name in required if not (self.results_dir / name).exists()]
        if missing:
            raise RuntimeError(f"Missing required artifacts: {missing}")
        audit = {
            "dataset": "STL-10", "seed": self.cfg.seed, "selected_test_images": len(self.subset_manifest),
            "balanced_test_counts": self.subset_manifest.groupby("class_name").size().to_dict(),
            "accepted_cue_conflicts": len(self.accepted), "cue_conflicts_per_pair_direction": self.cue_quota,
            "models": self.all_models, "required_artifacts": required,
        }
        self._save_json(audit, self.results_dir / "completion_audit.json")


def _load_rejections(path):
    return json.loads(Path(path).read_text()) if path else {}


def main():
    parser = argparse.ArgumentParser(description="Run ATML PA1 Task 1 on STL-10")
    parser.add_argument("--stage", choices=["pre-qc", "post-qc", "all"], default="pre-qc")
    parser.add_argument("--manual-rejections-json")
    parser.add_argument("--approve-visual-qc", action="store_true")
    args = parser.parse_args()

    pipeline = Task1Pipeline()
    if pipeline.device.type != "cuda":
        print("WARNING: CUDA is unavailable; execution will be very slow.")
    pipeline.prepare_data()
    pipeline.train_heads()
    if args.stage in {"pre-qc", "all"}:
        _, _, counts, sheets = pipeline.generate_cue_conflicts(_load_rejections(args.manual_rejections_json))
        print(counts.to_string(index=False))
        print("Inspect every QC sheet before evaluation:", *sheets, sep="\n")
        if args.stage == "pre-qc" or not args.approve_visual_qc:
            print("Stopped before predictions. Rerun with --stage post-qc --approve-visual-qc after review.")
            return
    if not args.approve_visual_qc:
        raise SystemExit("Post-QC evaluation requires --approve-visual-qc.")
    pipeline.load_accepted_conflicts()
    pipeline.extract_evaluation_features()
    results = pipeline.evaluate()
    for name, value in results.items():
        if isinstance(value, pd.DataFrame):
            print(f"\n{name}\n{value.to_string(index=False)}")
    print(f"Complete. Results saved to {pipeline.results_dir}")


if __name__ == "__main__":
    main()
