"""CNN 네트워크 — 1D 컨볼루션으로 시계열 특성 추출."""

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("cnn")
class CNNNetwork(BaseNetwork):

    def __init__(self, input_dim: int, num_filters: int = 64,
                 kernel_size: int = 3, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.num_filters = num_filters
        self._output_dim = num_filters

        # TODO: Conv1d 레이어 구성
        # TODO: 풀링, 배치 정규화 등 필요 시 추가

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x shape (batch, seq_len, input_dim) → 전치 → Conv1d → 특성 벡터
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim
