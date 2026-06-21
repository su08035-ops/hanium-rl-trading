"""DQN (Deep Q-Network) 에이전트.

epsilon-greedy 탐험 + 리플레이 버퍼 + 타깃 네트워크.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("dqn")
class DQNAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 1e-3,
                 gamma: float = 0.99, device: str = "cpu",
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, target_update: int = 10,
                 replay_size: int = 10000, **kwargs):
        super().__init__(network, lr, gamma, device, **kwargs)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        # TODO: 타깃 네트워크 초기화
        # TODO: 리플레이 버퍼 초기화 (replay_size)
        # TODO: 옵티마이저 초기화

    def select_action(self, state, explore: bool = True) -> int:
        # TODO: epsilon-greedy 행동 선택
        raise NotImplementedError

    def train_step(self, batch: dict) -> dict:
        # TODO: 리플레이 버퍼에서 샘플링 → Q-learning 업데이트
        raise NotImplementedError

    def save(self, path: Path) -> None:
        # TODO: 네트워크 + 옵티마이저 상태 저장
        raise NotImplementedError

    def load(self, path: Path) -> None:
        # TODO: 체크포인트 로드
        raise NotImplementedError
