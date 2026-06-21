"""재현성을 위한 랜덤 시드 고정."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """모든 난수 생성기의 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
