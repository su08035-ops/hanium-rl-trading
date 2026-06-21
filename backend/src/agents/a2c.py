"""A2C (Advantage Actor-Critic) 에이전트.

Actor-Critic 구조로 정책(actor)과 가치(critic)를 동시 학습.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("a2c")
class A2CAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 7e-4,
                 gamma: float = 0.99, device: str = "cpu",
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 **kwargs):
        super().__init__(network, lr, gamma, device, **kwargs)
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        # TODO: Actor head, Critic head 초기화
        # TODO: 옵티마이저 초기화

    def select_action(self, state, explore: bool = True) -> int:
        # TODO: 정책 네트워크에서 확률 분포 → 샘플링
        raise NotImplementedError

    def train_step(self, batch: dict) -> dict:
        # TODO: Advantage 계산 → Actor·Critic 동시 업데이트
        raise NotImplementedError

    def save(self, path: Path) -> None:
        # TODO: 체크포인트 저장
        raise NotImplementedError

    def load(self, path: Path) -> None:
        # TODO: 체크포인트 로드
        raise NotImplementedError
