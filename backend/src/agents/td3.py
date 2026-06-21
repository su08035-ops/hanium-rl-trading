"""TD3 (Twin Delayed DDPG) — 쌍둥이 비평가 + 지연 정책 업데이트.

포지션 유지/장기 보유 전략에 강함.
DDPG의 과대평가 문제를 쌍둥이 Critic과 지연 업데이트로 해결.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("td3")
class TD3Agent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 3e-4,
                 gamma: float = 0.99, device: str = "cpu",
                 tau: float = 0.005, policy_delay: int = 2,
                 noise_std: float = 0.2, noise_clip: float = 0.5,
                 replay_size: int = 100000, **kwargs):
        """
        Parameters
        ----------
        policy_delay : int
            Critic N번 업데이트마다 Actor 1번 업데이트.
        noise_std : float
            타깃 정책 스무딩 노이즈 표준편차.
        noise_clip : float
            노이즈 클리핑 범위.
        """
        super().__init__(network, lr, gamma, device, **kwargs)
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        # TODO: Actor 네트워크 (결정적 정책)
        # TODO: Critic 네트워크 2개 (쌍둥이 Q-함수)
        # TODO: 타깃 Actor, 타깃 Critic 네트워크
        # TODO: 리플레이 버퍼 초기화
        # TODO: 업데이트 카운터 (policy_delay 추적)

    def select_action(self, state, explore: bool = True) -> int:
        # TODO: Actor에서 결정적 행동 출력
        # TODO: explore=True이면 가우시안 노이즈 추가
        raise NotImplementedError

    def train_step(self, batch: dict) -> dict:
        # TODO: 1. 타깃 행동에 클리핑된 노이즈 추가 (스무딩)
        # TODO: 2. 쌍둥이 Critic 업데이트 (min Q 타깃)
        # TODO: 3. policy_delay마다 Actor 업데이트
        # TODO: 4. 타깃 네트워크 소프트 업데이트 (tau)
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError
