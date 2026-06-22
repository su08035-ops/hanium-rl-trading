"""학습 로그 유틸리티."""

import json
import logging
from pathlib import Path
from typing import Any, Dict


def setup_logger(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """파일 + 콘솔 핸들러를 가진 로거를 생성한다."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 중복 핸들러 방지
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 파일 핸들러
    fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def log_metrics(path: Path, episode: int, metrics: Dict[str, Any]) -> None:
    """에피소드별 메트릭을 JSON Lines 형식으로 기록한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"episode": episode, **metrics}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
