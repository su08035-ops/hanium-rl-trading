"""Mamba (Selective State Space Model) — 선형 시간 복잡도의 시퀀스 모델.

2026년 벤치마크 4위 (긴 시계열 / 고빈도 최고).
Transformer의 O(n²) 어텐션을 O(n) 선형 재귀로 대체.
입력에 따라 선택적으로 정보를 기억/망각하는 SSM(State Space Model).

구조:
  입력 → [선형 투영] → [Conv1D] → [선택적 SSM] → [게이팅] → 출력
"""

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("mamba")
class MambaNetwork(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 state_dim: int = 16, num_layers: int = 2,
                 conv_kernel: int = 4, expand_factor: int = 2,
                 dropout: float = 0.1, **kwargs):
        """
        Parameters
        ----------
        state_dim : int
            SSM의 상태 차원 (N).
        conv_kernel : int
            로컬 컨볼루션 커널 크기.
        expand_factor : int
            내부 차원 확장 배율.
        """
        super().__init__(input_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        self._output_dim = hidden_dim

        # TODO: Input Projection
        #   - input_dim → hidden_dim

        # TODO: Mamba Block × num_layers, 각 블록:
        #   1. Linear (hidden_dim → inner_dim * 2)  [분기: x, z]
        #   2. Conv1D (inner_dim, kernel=conv_kernel)  [로컬 문맥]
        #   3. SSM 파라미터 생성: B, C, Δ (입력 의존적 = selective)
        #   4. Selective Scan (이산화된 상태 공간 재귀)
        #   5. Gate: output = SSM(x) * SiLU(z)
        #   6. Linear (inner_dim → hidden_dim)
        #   7. Residual + LayerNorm

        # TODO: Output Norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x shape (batch, seq_len, input_dim)
        # 1. Input Projection
        # 2. Mamba Block × num_layers
        # 3. 마지막 시점의 hidden state 반환 (batch, hidden_dim)
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim
