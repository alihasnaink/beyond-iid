from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from shared.pacs import PACSLabeledDataset


def make_transforms(config):
    weights_transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    normalize = transforms.Normalize(mean=weights_transform.mean, std=weights_transform.std)
    train_transform = transforms.Compose([
        transforms.Resize((config["image_resize"], config["image_resize"])),
        transforms.RandomCrop(config["image_crop"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((config["image_resize"], config["image_resize"])),
        transforms.CenterCrop(config["image_crop"]),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_source_loaders(protocol, config):
    """Build source-only loaders. This function has no target-domain argument."""
    train_transform, eval_transform = make_transforms(config)
    source_loaders, val_datasets = {}, {}
    for offset, domain in enumerate(config["source_domains"]):
        generator = torch.Generator().manual_seed(config["seed"] + offset)
        dataset = PACSLabeledDataset(protocol["train"][domain], train_transform)
        source_loaders[domain] = DataLoader(
            dataset,
            batch_size=config["source_batch_per_domain"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
        val_datasets[domain] = PACSLabeledDataset(protocol["val"][domain], eval_transform)
    return source_loaders, val_datasets, eval_transform

