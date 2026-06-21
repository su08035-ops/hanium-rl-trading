"""TFT (Temporal Fusion Transformer) — LSTM + Attention + 변수 선택.

2026년 벤치마크 1위 (샤프 비율 2.27).
어떤 변수가 중요한지 자동으로 선택하고 해석 가능한 예측을 제공한다.

구조:
  입력 → [Variable Selection Network] → [LSTM Encoder] → [Multi-Head Attention] → 출력
         (어떤 지표가 중요한지)         (시간 순서 학습)    (중요 시점 집중)
"""

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("tft")
class TFTNetwork(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self._output_dim = hidden_dim

        # TODO: Variable Selection Network (VSN)
        #   - 각 입력 변수의 중요도를 GRN(Gated Residual Network)으로 학습
        #   - softmax로 변수별 가중치 산출

        # TODO: LSTM Encoder
        #   - 과거 시계열을 순차적으로 인코딩

        # TODO: Gated Layer Normalization (GLN)
        #   - LSTM 출력에 게이트 적용

        # TODO: Multi-Head Attention
        #   - 인코딩된 시계열에서 중요 시점에 집중

        # TODO: Position-wise Feed Forward
        #   - 최종 특성 벡터 생성

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x shape (batch, seq_len, input_dim)
        # 1. Variable Selection → 중요 변수 가중 합산
        # 2. LSTM Encoder → 시계열 인코딩
        # 3. Multi-Head Attention → 중요 시점 집중
        # 4. 최종 특성 벡터 반환 (batch, hidden_dim)
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim
