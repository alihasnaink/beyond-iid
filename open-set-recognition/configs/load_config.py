from pathlib import Path
import yaml


def load_config(task_dir: Path, method: str):
    with (task_dir / "configs/base.yaml").open() as stream:
        config = yaml.safe_load(stream)
    with (task_dir / f"configs/{method}.yaml").open() as stream:
        config.update(yaml.safe_load(stream))
    return config
