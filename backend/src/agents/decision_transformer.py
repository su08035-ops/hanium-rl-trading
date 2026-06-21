"""Decision Transformer — RL을 시퀀스 예측 문제로 변환.

과거 거래 이력에서 바로 학습 (Offline RL).
실시간 시뮬레이션 없이 과거 데이터만으로 학습 가능.

원리:
  GPT처럼 (목표수익, 상태, 행동) 시퀀스를 입력받아
  다음 행동을 예측. 목표수익을 높게 설정하면 고수익 전략,
  낮게 설정하면 보수적 전략을 생성.

구조:
  [R̂₁, s₁, a₁, R̂₂, s₂, a₂, ...] → GPT-style Transformer → â_next
"""

from pathlib import Path

import torch
import torch.nn as nn

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("decision_transformer")
class DecisionTransformerAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 1e-4,
                 gamma: float = 0.99, device: str = "cpu",
                 context_len: int = 20, target_return: float = 0.3,
                 n_heads: int = 4, n_layers: int = 3,
                 dropout: float = 0.1, **kwargs):
        """
        Parameters
        ----------
        context_len : int
            참조할 과거 시퀀스 길이 (timesteps).
        target_return : float
            목표 수익률 (높을수록 공격적, 낮을수록 보수적).
        """
        super().__init__(network, lr, gamma, device, **kwargs)
        self.context_len = context_len
        self.target_return = target_return
        # TODO: Return 임베딩 (Linear)
        # TODO: State 임베딩 (network 활용)
        # TODO: Action 임베딩 (Embedding 또는 Linear)
        # TODO: Timestep 임베딩 (Embedding)
        # TODO: GPT-style Transformer (Causal, n_layers)
        # TODO: Action 예측 헤드

        # TODO: 학습 데이터 구성
        #   과거 거래 이력에서 (return-to-go, state, action) 시퀀스 생성

    def select_action(self, state, explore: bool = True) -> int:
        # TODO: 현재까지의 시퀀스 + target_return으로 다음 행동 예측
        # TODO: Transformer로 autoregressive 생성
        raise NotImplementedError

    def train_step(self, batch: dict) -> dict:
        # TODO: Supervised learning 방식
        # TODO: 시퀀스 입력 → 행동 예측 → Cross-entropy loss
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError
