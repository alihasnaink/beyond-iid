from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_linear_head(train_x, train_y, val_x, val_y, feature_dim, model_name, cfg, device, n_classes=10):
    seed_everything(cfg.seed)
    head = nn.Linear(feature_dim, n_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.head_lr, weight_decay=cfg.head_weight_decay)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True, generator=generator)
    val_x, val_y = val_x.to(device), val_y.to(device)
    best_accuracy, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(1, cfg.max_head_epochs + 1):
        head.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            logits = head(val_x)
            val_loss = F.cross_entropy(logits, val_y).item()
            val_accuracy = (logits.argmax(1) == val_y).float().mean().item()
        history.append({"model": model_name, "epoch": epoch, "val_loss": val_loss, "val_accuracy": val_accuracy})
        if val_accuracy > best_accuracy + 1e-12:
            best_accuracy, stale = val_accuracy, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
            if stale >= cfg.head_patience:
                break
    head.load_state_dict(best_state)
    return head.cpu().eval(), pd.DataFrame(history)


@torch.inference_mode()
def predict_head(head, features):
    return head(features.float()).softmax(1)
