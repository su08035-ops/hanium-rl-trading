"""PPO (Proximal Policy Optimization) 에이전트.

클리핑 기반 정책 업데이트 + GAE (Generalized Advantage Estimation).
"""

from pathlib import Path

import torch
import torch.nn as nn

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("ppo")
class PPOAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 3e-4,
                 gamma: float = 0.99, device: str = "cpu",
                 gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 ppo_epochs: int = 4, rollout_steps: int = 2048,
                 **kwargs):
        super().__init__(network, lr, gamma, device, **kwargs)
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.rollout_steps = rollout_steps
        # TODO: Actor head, Critic head 초기화
        # TODO: 옵티마이저 초기화
        # TODO: 롤아웃 버퍼 초기화

    def select_action(self, state, explore: bool = True) -> int:
        # TODO: 정책에서 행동 샘플링 + log_prob 저장
        raise NotImplementedError

    def train_step(self, batch: dict) -> dict:
        # TODO: GAE 계산 → 클리핑 PPO 업데이트 (ppo_epochs 반복)
        raise NotImplementedError

    def save(self, path: Path) -> None:
        # TODO: 체크포인트 저장
        raise NotImplementedError

    def load(self, path: Path) -> None:
        # TODO: 체크포인트 로드
        raise NotImplementedError
