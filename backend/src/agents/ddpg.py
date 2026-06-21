"""DDPG (Deep Deterministic Policy Gradient) — 연속 행동 공간 Actor-Critic.

포트폴리오 비중 조절 등 연속적 판단에 적합.
TD3, SAC의 기반이 되는 알고리즘으로 베이스라인 비교에 사용.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("ddpg")
class DDPGAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 1e-3,
                 gamma: float = 0.99, device: str = "cpu",
                 tau: float = 0.005, noise_std: float = 0.1,
                 replay_size: int = 100000, **kwargs):
        super().__init__(network, lr, gamma, device, **kwargs)
        self.tau = tau
        self.noise_std = noise_std
        # TODO: Actor 네트워크 (결정적 정책)
        # TODO: Critic 네트워크 (Q-함수)
        # TODO: 타깃 Actor, 타깃 Critic
        # TODO: 리플레이 버퍼 초기화
        # TODO: Ornstein-Uhlenbeck 노이즈 (또는 가우시안)

    def select_action(self, state, explore: bool = True) -> int:
        # TODO: Actor 출력 + 탐험 노이즈
        raise NotImplementedError

    def train_step(self, batch: dict) -> dict:
        # TODO: 1. Critic 업데이트 (MSE loss)
        # TODO: 2. Actor 업데이트 (Critic을 통한 정책 기울기)
        # TODO: 3. 타깃 네트워크 소프트 업데이트
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError
