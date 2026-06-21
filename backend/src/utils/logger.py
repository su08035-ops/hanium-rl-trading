"""학습 로그 유틸리티."""

import json
import logging
from pathlib import Path
from typing import Any, Dict


def setup_logger(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """파일 + 콘솔 핸들러를 가진 로거를 생성한다.

    Parameters
    ----------
    name : str
        로거 이름.
    log_dir : Path
        로그 파일 저장 디렉토리.
    level : int
        로깅 레벨.

    Returns
    -------
    logging.Logger
    """
    # TODO: log_dir 생성
    # TODO: 파일 핸들러 (train.log)
    # TODO: 콘솔 핸들러
    # TODO: 포맷 설정
    raise NotImplementedError


def log_metrics(path: Path, episode: int, metrics: Dict[str, Any]) -> None:
    """에피소드별 메트릭을 JSON Lines 형식으로 기록한다."""
    # TODO: {"episode": ..., **metrics} 한 줄 append
    raise NotImplementedError
