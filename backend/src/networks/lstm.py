"""LSTM 네트워크 — 시계열 패턴 학습용."""

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("lstm")
class LSTMNetwork(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._output_dim = hidden_dim

        # TODO: nn.LSTM 초기화
        # TODO: 드롭아웃, 레이어 정규화 등 필요 시 추가

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x shape (batch, seq_len, input_dim) → LSTM → 마지막 hidden state 반환
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim
