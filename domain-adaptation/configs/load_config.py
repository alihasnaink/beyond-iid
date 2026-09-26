from __future__ import annotations

from pathlib import Path
import yaml


def load_config(task_dir: Path, method: str, overrides=None):
    with (task_dir / "configs" / "base.yaml").open() as stream:
        config = yaml.safe_load(stream)
    with (task_dir / "configs" / f"{method}.yaml").open() as stream:
        config.update(yaml.safe_load(stream))
    if overrides:
        config.update(overrides)
    return config
