from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16
from tqdm.auto import tqdm


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def normalize_batch(x: torch.Tensor, family: str) -> torch.Tensor:
    mean, std = (CLIP_MEAN, CLIP_STD) if family == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
    return (x - mean.to(x.device)) / std.to(x.device)


class ResNetFeatures(nn.Module):
    family, dim = "imagenet", 2048

    def __init__(self):
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.features(normalize_batch(x, self.family)).flatten(1)


class ViTFeatures(nn.Module):
    family, dim = "imagenet", 768

    def __init__(self):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    def forward(self, x):
        x = self.model._process_input(normalize_batch(x, self.family))
        class_token = self.model.class_token.expand(x.shape[0], -1, -1)
        return self.model.encoder(torch.cat([class_token, x], dim=1))[:, 0]


class CLIPFeatures(nn.Module):
    family, dim = "clip", 512

    def __init__(self):
        super().__init__()
        import open_clip

        self.model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

    def forward(self, x):
        return self.model.encode_image(normalize_batch(x, self.family), normalize=True)


MODEL_BUILDERS = {"resnet50": ResNetFeatures, "vit_b16": ViTFeatures, "clip_vit_b32": CLIPFeatures}


def build_backbone(name: str, device: torch.device) -> nn.Module:
    model = MODEL_BUILDERS[name]().eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def extract_features(model, dataset, device, batch_size=64, num_workers=2, desc="features"):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features, labels, identifiers = [], [], []
    for images, target, item_ids in tqdm(loader, desc=desc, leave=False):
        context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with context:
            feature = model(images.to(device, non_blocking=True))
        features.append(feature.float().cpu())
        labels.append(torch.as_tensor(target).long().cpu())
        identifiers.extend(torch.as_tensor(item_ids).tolist())
    return torch.cat(features), torch.cat(labels), identifiers
