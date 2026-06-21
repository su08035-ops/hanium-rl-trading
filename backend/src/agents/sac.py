"""SAC (Soft Actor-Critic) — 엔트로피 최대화 기반 Actor-Critic.

단일 알고리즘 중 벤치마크 1위.
보상 최대화와 동시에 탐험(엔트로피)도 최대화하여
하이퍼파라미터에 강건하고 안정적인 학습을 제공한다.
연속 행동 공간(포트폴리오 비중 조절 등)에 최적.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("sac")
class SACAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 3e-4,
                 gamma: float = 0.99, device: str = "cpu",
                 tau: float = 0.005, alpha: float = 0.2,
                 auto_alpha: bool = True, replay_size: int = 100000,
                 **kwargs):
        """
        Parameters
        ----------
        tau : float
            타깃 네트워크 소프트 업데이트 비율.
        alpha : float
            엔트로피 계수 (auto_alpha=True이면 자동 조절).
        auto_alpha : bool
            True이면 엔트로피 계수를 자동으로 학습.
        """
        super().__init__(network, lr, gamma, device, **kwargs)
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        # TODO: Actor 네트워크 (정책 — 가우시안 분포 출력)
        # TODO: Critic 네트워크 2개 (쌍둥이 Q-함수)
        # TODO: 타깃 Critic 네트워크 2개
        # TODO: 리플레이 버퍼 초기화 (replay_size)
        # TODO: alpha 자동 조절 시 log_alpha 파라미터 + 옵티마이저

    def select_action(self, state, explore: bool = True) -> int:
        # TODO: Actor에서 가우시안 분포 → 샘플링 (explore=True)
        # TODO: explore=False이면 평균값 사용 (deterministic)
        raise NotImplementedError

    def train_step(self, batch: dict) -> dict:
        # TODO: 1. 쌍둥이 Critic 업데이트 (min Q 사용)
        # TODO: 2. Actor 업데이트 (Q + alpha * entropy 최대화)
        # TODO: 3. auto_alpha이면 alpha 업데이트
        # TODO: 4. 타깃 네트워크 소프트 업데이트 (tau)
        raise NotImplementedError

    def save(self, path: Path) -> None:
        # TODO: Actor, Critic, alpha 상태 저장
        raise NotImplementedError

    def load(self, path: Path) -> None:
        # TODO: 체크포인트 로드
        raise NotImplementedError
