"""IQN (Implicit Quantile Networks) — 수익 분포를 학습하는 분포형 RL.

리스크 관리 최고 — CVaR(조건부 VaR)를 직접 최적화 가능.
위험 성향을 조절할 수 있어 보수적/공격적 전략 모두 지원.

구조:
  상태 → [네트워크] → 수익 분포 전체를 학습
  → "평균적으로 얼마 벌까" 뿐 아니라 "최악의 경우 얼마 잃을까"도 판단
"""

from pathlib import Path

import torch
import torch.nn as nn

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("iqn")
class IQNAgent(BaseAgent):

    def __init__(self, network: nn.Module, lr: float = 5e-5,
                 gamma: float = 0.99, device: str = "cpu",
                 n_quantiles: int = 64, embedding_dim: int = 64,
                 kappa: float = 1.0, cvar_alpha: float = 1.0,
                 replay_size: int = 100000, **kwargs):
        """
        Parameters
        ----------
        n_quantiles : int
            샘플링할 분위수 개수.
        embedding_dim : int
            분위수 임베딩 차원.
        kappa : float
            Huber loss의 임계값.
        cvar_alpha : float
            CVaR 신뢰 수준 (1.0=위험 중립, 0.25=매우 보수적).
        """
        super().__init__(network, lr, gamma, device, **kwargs)
        self.n_quantiles = n_quantiles
        self.embedding_dim = embedding_dim
        self.kappa = kappa
        self.cvar_alpha = cvar_alpha
        # TODO: 분위수 임베딩 네트워크 (cos 기반)
        #   τ → cos(πiτ) → Linear → 임베딩
        # TODO: Q-네트워크 (상태 특성 × 분위수 임베딩 → 분위수별 Q값)
        # TODO: 타깃 네트워크
        # TODO: 리플레이 버퍼

    def select_action(self, state, explore: bool = True) -> int:
        # TODO: τ를 [0, cvar_alpha] 범위에서 샘플링 (위험 조절)
        # TODO: 각 행동의 분위수 평균 → Q값 → argmax
        raise NotImplementedError

    def train_step(self, batch: dict) -> dict:
        # TODO: 1. τ, τ' 독립 샘플링
        # TODO: 2. 분위수 Huber 회귀 loss 계산
        # TODO: 3. 옵티마이저 업데이트
        # TODO: 4. 타깃 네트워크 주기적 업데이트
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError
