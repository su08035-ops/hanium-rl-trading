"""DNN (Deep Neural Network) — 완전연결 피드포워드 네트워크."""

from typing import List

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("dnn")
class DNN(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 **kwargs):
        super().__init__(input_dim, **kwargs)
        hidden_dims = hidden_dims or [128, 64]
        self._output_dim = hidden_dims[-1]

        # TODO: hidden_dims에 따른 Linear + ReLU 레이어 구성
        # 예: input_dim → 128 → ReLU → 64 → ReLU
        self.layers = nn.Sequential()  # TODO: 레이어 채우기

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: self.layers를 통과시켜 특성 벡터 반환
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim
