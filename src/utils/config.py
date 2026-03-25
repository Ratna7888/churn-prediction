"""Load and access YAML configuration."""

from pathlib import Path
import yaml


def load_config(path: str = "configs/model_config.yaml") -> dict:
    """Load YAML config file and return as dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)