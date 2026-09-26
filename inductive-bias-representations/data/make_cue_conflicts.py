from __future__ import annotations

import gc
import math
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from data.transforms import canonical_tensor


VGG_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
DECODER_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"


def _conv_block(in_channels, out_channels, relu=True):
    layers = [nn.ReflectionPad2d(1), nn.Conv2d(in_channels, out_channels, 3)]
    if relu:
        layers.append(nn.ReLU())
    return layers


def make_decoder():
    layers = _conv_block(512, 256) + [nn.Upsample(scale_factor=2, mode="nearest")]
    layers += _conv_block(256, 256) + _conv_block(256, 256) + _conv_block(256, 256)
    layers += _conv_block(256, 128) + [nn.Upsample(scale_factor=2, mode="nearest")]
    layers += _conv_block(128, 128) + _conv_block(128, 64) + [nn.Upsample(scale_factor=2, mode="nearest")]
    layers += _conv_block(64, 64) + _conv_block(64, 3, relu=False)
    return nn.Sequential(*layers)


def make_vgg():
    layers = [nn.Conv2d(3, 3, 1)]
    layers += _conv_block(3, 64) + _conv_block(64, 64) + [nn.MaxPool2d(2, 2, ceil_mode=True)]
    layers += _conv_block(64, 128) + _conv_block(128, 128) + [nn.MaxPool2d(2, 2, ceil_mode=True)]
    layers += _conv_block(128, 256) + _conv_block(256, 256) + _conv_block(256, 256) + _conv_block(256, 256)
    layers += [nn.MaxPool2d(2, 2, ceil_mode=True)] + _conv_block(256, 512)
    return nn.Sequential(*layers)


def _download_if_missing(url: str, path: Path):
    if not path.exists():
        print(f"Downloading {path.name} ...")
        urllib.request.urlretrieve(url, path)


def _load_vgg_encoder_state(vgg: nn.Module, checkpoint):
    """Load the relu4_1 encoder prefix from the released full-VGG checkpoint."""
    expected = vgg.state_dict()
    encoder_state = {key: value for key, value in checkpoint.items() if key in expected}
    missing = sorted(set(expected) - set(encoder_state))
    if missing:
        raise RuntimeError(f"AdaIN VGG checkpoint is missing encoder keys: {missing}")
    # Strict loading after filtering still validates every required key and tensor shape.
    vgg.load_state_dict(encoder_state, strict=True)


def load_adain(cache_dir: Path, device):
    vgg_path, decoder_path = cache_dir / "vgg_normalised.pth", cache_dir / "decoder.pth"
    _download_if_missing(VGG_URL, vgg_path)
    _download_if_missing(DECODER_URL, decoder_path)
    vgg, decoder = make_vgg(), make_decoder()
    _load_vgg_encoder_state(vgg, torch.load(vgg_path, map_location="cpu", weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))
    for module in (vgg, decoder):
        module.eval().to(device)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return vgg, decoder


def adaptive_instance_normalization(content, style, eps=1e-5):
    content_mean = content.mean((2, 3), keepdim=True)
    content_std = content.var((2, 3), keepdim=True, unbiased=False).add(eps).sqrt()
    style_mean = style.mean((2, 3), keepdim=True)
    style_std = style.var((2, 3), keepdim=True, unbiased=False).add(eps).sqrt()
    return (content - content_mean) / content_std * style_std + style_mean


@torch.inference_mode()
def stylize(vgg, decoder, content, style, alpha):
    content_features, style_features = vgg(content), vgg(style)
    target = adaptive_instance_normalization(content_features, style_features)
    target = alpha * target + (1 - alpha) * content_features
    return decoder(target).clamp(0, 1)


def automated_qc(image: torch.Tensor):
    reasons = []
    if not torch.isfinite(image).all():
        reasons.append("non-finite")
    if image.std().item() < 0.035:
        reasons.append("near-uniform")
    clipped = ((image < 0.01) | (image > 0.99)).float().mean().item()
    if clipped > 0.50:
        reasons.append("excessive-clipping")
    return reasons, clipped


def generate_candidates(test_dataset, subset_manifest, class_names, cue_pairs, cfg, cache_dir, output_dir, device, manual_rejections=None):
    manual_rejections = manual_rejections or {}
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    eval_by_class = {
        index: subset_manifest.loc[subset_manifest.class_index == index, "official_test_index"].tolist()
        for index in range(len(class_names))
    }
    vgg, decoder = load_adain(cache_dir, device)
    rows = []
    try:
        for pair_id, (class_a, class_b) in enumerate(cue_pairs):
            for content_name, style_name in [(class_a, class_b), (class_b, class_a)]:
                content_label, style_label = class_to_idx[content_name], class_to_idx[style_name]
                rng = np.random.default_rng(cfg.seed + pair_id * 100 + content_label)
                content_ids = rng.choice(eval_by_class[content_label], cfg.conflicts_per_direction, replace=False)
                style_ids = rng.choice(eval_by_class[style_label], cfg.conflicts_per_direction, replace=False)
                for rank, (content_id, style_id) in enumerate(zip(content_ids, style_ids)):
                    conflict_id = f"p{pair_id}_{content_name}_shape__{style_name}_texture_{rank:02d}"
                    path = output_dir / f"{conflict_id}.png"
                    content = canonical_tensor(test_dataset[int(content_id)][0], cfg.image_size).unsqueeze(0).to(device)
                    style = canonical_tensor(test_dataset[int(style_id)][0], cfg.image_size).unsqueeze(0).to(device)
                    output = stylize(vgg, decoder, content, style, cfg.style_alpha).squeeze(0).cpu()
                    TF.to_pil_image(output).save(path)
                    reasons, clipped = automated_qc(output)
                    if conflict_id in manual_rejections:
                        reasons.append("manual: " + manual_rejections[conflict_id])
                    rows.append({
                        "conflict_id": conflict_id, "pair_id": pair_id,
                        "content_class": content_name, "content_label": content_label,
                        "style_class": style_name, "style_label": style_label,
                        "content_official_index": int(content_id), "style_official_index": int(style_id),
                        "path": str(path), "pixel_std": float(output.std()), "clipped_fraction": clipped,
                        "rejection_reason": "; ".join(reasons), "prebalance_valid": not reasons,
                    })
                    del content, style, output
    finally:
        del vgg, decoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def finalize_candidates(candidates, subset_manifest, cfg):
    valid = candidates[candidates.prebalance_valid].copy()
    cell_columns = ["pair_id", "content_class", "style_class"]
    quota = min(cfg.conflicts_per_direction, int(valid.groupby(cell_columns).size().min()))
    if quota < cfg.min_conflicts_per_direction:
        raise RuntimeError(f"Weakest direction has {quota} valid conflicts; need {cfg.min_conflicts_per_direction}.")
    accepted = valid.sort_values("conflict_id").groupby(cell_columns, group_keys=False).head(quota).copy()
    accepted["accepted"] = True
    accepted["content_eval_position"] = accepted.content_official_index.map(
        dict(zip(subset_manifest.official_test_index, subset_manifest.eval_position))
    )
    if len(accepted) < 200 or accepted.groupby(cell_columns).size().nunique() != 1:
        raise RuntimeError("Accepted conflicts are not sufficiently numerous and balanced.")
    counts = pd.DataFrame({
        "candidates": [len(candidates)],
        "automated_or_manual_rejected": [(~candidates.prebalance_valid).sum()],
        "valid_before_balancing": [len(valid)],
        "accepted_after_balancing": [len(accepted)],
        "per_direction_quota": [quota],
    })
    return accepted, counts, quota


def save_qc_sheets(candidates, results_dir: Path, sheet_size=50):
    paths = []
    for sheet_number, start in enumerate(range(0, len(candidates), sheet_size), start=1):
        sheet = candidates.iloc[start:start + sheet_size]
        # Keep the saved image close to its notebook display size.  The previous
        # 2700x4860 render needed several large RGBA buffers per sheet and could
        # exhaust a Kaggle kernel while savefig/display held those buffers.
        fig, axes = plt.subplots(10, 5, figsize=(12, 20))
        for axis in axes.flat:
            axis.axis("off")
        for axis, row in zip(axes.flat, sheet.itertuples()):
            with Image.open(row.path) as image:
                axis.imshow(image.copy())
            axis.set_title(f"{row.conflict_id}\nshape={row.content_class}; texture={row.style_class}", fontsize=6)
        fig.suptitle(f"Cue-conflict visual QC - sheet {sheet_number}", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        path = results_dir / f"cue_conflict_qc_contact_sheet_{sheet_number:02d}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        del fig, axes
        gc.collect()
        paths.append(path)
    return paths


class ConflictDataset(Dataset):
    def __init__(self, manifest, image_size=224):
        self.manifest = manifest.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]
        with Image.open(row.path) as image:
            tensor = canonical_tensor(image, self.image_size)
        return tensor, int(row.content_label), index
