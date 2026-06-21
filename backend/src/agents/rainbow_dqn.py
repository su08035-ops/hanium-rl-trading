"""Rainbow DQN — DQN의 6가지 개선을 모두 합친 강화 버전.

이산 행동 공간 알고리즘 중 최고 성능.
기존 DQN 대비 32%+ 성능 향상.

포함된 기법:
  1. Double Q-learning (과대평가 방지)
  2. Prioritized Experience Replay (중요 경험 우선 학습)
  3. Dueling Network (가치/이점 분리)
  4. Multi-step Returns (n-step 부트스트래핑)
  5. Distributional RL / C51 (수익 분포 학습)
  6. Noisy Networks (파라미터 노이즈로 탐험)
"""

from pathlib import Path

import torch
import torch.nn as nn

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("rainbow")
class RainbowDQNAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 6.25e-5,
                 gamma: float = 0.99, device: str = "cpu",
                 n_step: int = 3, n_atoms: int = 51,
                 v_min: float = -10.0, v_max: float = 10.0,
                 replay_size: int = 100000, priority_alpha: float = 0.6,
                 priority_beta: float = 0.4, **kwargs):
        """
        Parameters
        ----------
        n_step : int
            멀티스텝 리턴의 스텝 수.
        n_atoms : int
            C51 분포의 원자 수.
        v_min, v_max : float
            수익 분포의 최소/최대 범위.
        priority_alpha : float
            우선순위 리플레이의 alpha (우선순위 강도).
        priority_beta : float
            중요도 샘플링 보정 계수.
        """
        super().__init__(network, lr, gamma, device, **kwargs)
        self.n_step = n_step
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        # TODO: Dueling + Noisy + Distributional 네트워크 구성
        # TODO: 타깃 네트워크 초기화
        # TODO: Prioritized Replay Buffer 초기화
        # TODO: 분포 서포트 벡터 생성: linspace(v_min, v_max, n_atoms)

    def select_action(self, state, explore: bool = True) -> int:
        # TODO: Noisy Network이 탐험을 담당 (epsilon 불필요)
        # TODO: 분포의 기대값으로 Q값 계산 → argmax
        raise NotImplementedError

    def train_step(self, batch: dict) -> dict:
        # TODO: 1. 우선순위 리플레이에서 샘플링
        # TODO: 2. n-step 타깃 분포 계산
        # TODO: 3. KL-divergence 또는 cross-entropy loss
        # TODO: 4. 우선순위 업데이트
        # TODO: 5. 타깃 네트워크 주기적 업데이트
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError
