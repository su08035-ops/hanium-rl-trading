"""주식 매매 환경 — gymnasium 스타일.

상태: 과거 window_size일의 OHLCV + 기술지표 + 포지션 정보
행동: 0=매수, 1=매도, 2=관망 (이산)
보상: 포트폴리오 가치 변화율 (또는 로그 수익률)
"""

from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_balance: int = 10_000_000,
                 commission: float = 0.00015, window_size: int = 20):
        """
        Parameters
        ----------
        df : pd.DataFrame
            전처리 완료된 시세 데이터.
        initial_balance : int
            초기 자본금.
        commission : float
            매매 수수료율.
        window_size : int
            상태에 포함할 과거 일수.
        """
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size

        self.action_space = gym.spaces.Discrete(3)
        # TODO: observation_space 정의
        self.observation_space = None

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """환경을 초기 상태로 리셋한다."""
        # TODO: 잔고·포지션·현재 스텝 초기화
        # TODO: 초기 관측값 반환
        raise NotImplementedError

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """행동을 실행하고 다음 상태·보상을 반환한다."""
        # TODO: 매수/매도/관망 실행
        # TODO: 수수료 반영
        # TODO: 보상 계산
        # TODO: 종료 조건 확인
        raise NotImplementedError

    def _get_observation(self) -> np.ndarray:
        """현재 스텝의 관측값을 구성한다."""
        # TODO: window_size만큼의 과거 데이터 + 포지션 정보
        raise NotImplementedError

    def render(self, mode="human"):
        # TODO: 현재 포트폴리오 상태 출력
        pass
