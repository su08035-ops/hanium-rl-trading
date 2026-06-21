"""앙상블 에이전트 — 직전 구간 샤프 비율이 가장 높은 에이전트를 선택.

여러 학습된 에이전트를 보유하고, 일정 주기마다 최근 성과를 평가하여
가장 우수한 에이전트의 행동을 따른다.
"""

from pathlib import Path
from typing import List

import torch.nn as nn

from .base_agent import BaseAgent
from .registry import AgentRegistry


@AgentRegistry.register("ensemble")
class EnsembleAgent(BaseAgent):

    def __init__(self, network: nn.Module = None, lr: float = 0.0,
                 gamma: float = 0.99, device: str = "cpu",
                 agents: List[BaseAgent] = None,
                 eval_window: int = 20, **kwargs):
        super().__init__(network or nn.Identity(), lr, gamma, device, **kwargs)
        self.agents = agents or []
        self.eval_window = eval_window
        self.active_agent: BaseAgent | None = None
        # TODO: 각 에이전트의 최근 성과 추적 구조

    def add_agent(self, agent: BaseAgent) -> None:
        """앙상블에 에이전트를 추가한다."""
        # TODO: 에이전트 리스트에 추가
        raise NotImplementedError

    def _select_best_agent(self, recent_returns: dict) -> BaseAgent:
        """최근 구간 샤프 비율 기준으로 최고 에이전트를 선택한다."""
        # TODO: 샤프 비율 계산 → 최고 에이전트 반환
        raise NotImplementedError

    def select_action(self, state, explore: bool = True) -> int:
        # TODO: active_agent의 행동을 따른다
        raise NotImplementedError

    def train_step(self, batch: dict) -> dict:
        # TODO: 개별 에이전트 학습은 외부에서 수행, 여기서는 pass
        raise NotImplementedError

    def save(self, path: Path) -> None:
        # TODO: 모든 하위 에이전트 체크포인트 저장
        raise NotImplementedError

    def load(self, path: Path) -> None:
        # TODO: 모든 하위 에이전트 체크포인트 로드
        raise NotImplementedError
