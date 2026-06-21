"""YAML config 로드 + base 상속 병합."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """config 파일을 로드하고, extends가 있으면 base를 병합한다.

    Parameters
    ----------
    config_path : str
        config 파일 경로 (예: "configs/ppo_lstm.yaml").

    Returns
    -------
    dict
        병합된 설정 딕셔너리.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "extends" in cfg:
        base_path = config_path.parent / cfg.pop("extends")
        base_cfg = load_config(str(base_path))
        base_cfg = _deep_merge(base_cfg, cfg)
        return base_cfg

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """base dict에 override를 재귀적으로 병합한다."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
