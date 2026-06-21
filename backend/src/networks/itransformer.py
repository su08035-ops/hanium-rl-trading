"""iTransformer (Inverted Transformer) — 변수별 시계열을 토큰으로 취급.

2026년 벤치마크 5위 (다변량 예측 정확도).
기존 Transformer: 시점(time step)이 토큰
iTransformer: 변수(variate)가 토큰 → 변수 간 상관관계를 더 잘 포착.

구조:
  [변수1의 20일] [변수2의 20일] ... → 각각을 토큰으로 임베딩 → Attention → 출력
  (종가 시계열)  (RSI 시계열)       (변수 간 관계 학습)
"""

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("itransformer")
class ITransformerNetwork(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2,
                 seq_len: int = 20, dropout: float = 0.1, **kwargs):
        """
        Parameters
        ----------
        seq_len : int
            각 변수의 시계열 길이 (= window_size).
        """
        super().__init__(input_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self._output_dim = hidden_dim

        # TODO: Variate Embedding
        #   - 각 변수의 전체 시계열(seq_len,)을 하나의 토큰(hidden_dim,)으로 임베딩
        #   - Linear(seq_len, hidden_dim) 또는 MLP

        # TODO: Transformer Encoder
        #   - 토큰 = 변수, 어텐션 = 변수 간 관계 학습
        #   - nn.TransformerEncoderLayer × num_layers

        # TODO: Output Projection
        #   - 모든 변수 토큰을 집계하여 최종 특성 벡터 생성

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x shape (batch, seq_len, input_dim)
        # 1. 전치: (batch, input_dim, seq_len) — 각 변수의 시계열을 분리
        # 2. Variate Embedding: (batch, input_dim, hidden_dim) — 각 변수 시계열 → 토큰
        # 3. Transformer Encoder: 변수 간 어텐션
        # 4. 집계(평균 풀링 등) → (batch, hidden_dim)
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim
