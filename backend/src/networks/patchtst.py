"""PatchTST — 시계열을 패치(조각)로 잘라서 Transformer로 학습.

2026년 벤치마크 3위 (LSTM과 조합 시 꼬리 위험 관리 최고).
이미지의 ViT처럼, 시계열을 일정 길이의 패치로 분할하여 각 패치를
하나의 토큰으로 취급한다. 채널 독립(channel-independent) 설계.

구조:
  [20일 시계열] → [5일씩 4개 패치로 분할] → [패치 임베딩] → [Transformer] → 출력
"""

import torch
import torch.nn as nn

from .base_network import BaseNetwork
from .registry import NetworkRegistry


@NetworkRegistry.register("patchtst")
class PatchTSTNetwork(BaseNetwork):

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 patch_len: int = 5, stride: int = 5,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, **kwargs):
        """
        Parameters
        ----------
        patch_len : int
            각 패치의 길이 (일 수).
        stride : int
            패치 간 이동 간격.
        """
        super().__init__(input_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.patch_len = patch_len
        self.stride = stride
        self._output_dim = hidden_dim

        # TODO: Patch Embedding
        #   - (batch, seq_len, input_dim) → 패치로 분할
        #   - 각 패치를 Linear로 hidden_dim에 투영

        # TODO: Positional Encoding
        #   - 패치 순서 정보 추가

        # TODO: Transformer Encoder
        #   - nn.TransformerEncoderLayer × num_layers

        # TODO: Output Head
        #   - 패치 출력을 평균 풀링하여 최종 특성 벡터 생성

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x shape (batch, seq_len, input_dim)
        # 1. 시계열을 patch_len 단위로 분할
        # 2. 각 패치를 임베딩
        # 3. Positional Encoding 추가
        # 4. Transformer Encoder 통과
        # 5. 평균 풀링 → (batch, hidden_dim)
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim
