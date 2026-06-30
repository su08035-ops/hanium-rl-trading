"""주식 매매 환경 — gymnasium 스타일.

상태: 과거 window_size일의 OHLCV + 기술지표 + 포지션 정보
행동: 0=매수, 1=매도, 2=관망 (이산)
보상: 포트폴리오 가치 변화율
"""

from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    # 행동 상수
    BUY = 0
    SELL = 1
    HOLD = 2

    def __init__(self, df: pd.DataFrame, initial_balance: int = 10_000_000,
                 commission: float = 0.00015, window_size: int = 20,
                 trade_ratio: float = 1.0, raw_prices: np.ndarray = None,
                 theme_signal: dict = None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            전처리 완료된 시세 데이터 (정규화된 특성).
        initial_balance : int
            초기 자본금.
        commission : float
            매매 수수료율.
        window_size : int
            상태에 포함할 과거 일수.
        trade_ratio : float
            매수 시 잔고 대비 투자 비율 (1.0 = 전량).
        raw_prices : np.ndarray, optional
            정규화 전 원본 종가. None이면 df["close"] 사용.
        theme_signal : dict, optional
            {날짜(YYYYMMDD): bool} 테마 활성 신호.
            None이면 매일 매매 허용, 주어지면 True인 날만 매수 허용.
        """
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.trade_ratio = trade_ratio
        self.theme_signal = theme_signal

        # 날짜 인덱스 → 테마 활성 배열 변환
        if theme_signal is not None and hasattr(df, "index"):
            self._theme_active = self._build_theme_array(df, theme_signal)
        else:
            self._theme_active = None

        # 원본 종가 (매매 가격 계산용) — 정규화되지 않은 값
        self.prices = raw_prices if raw_prices is not None else df["close"].values

        # 상태에 사용할 특성 (모든 수치 컬럼)
        self.features = df.select_dtypes(include=[np.number]).values
        self.n_features = self.features.shape[1]

        # 관측 공간: (window_size, n_features + 3)
        #   +3 = [보유비율, 수익률, 잔고비율]
        obs_shape = (window_size, self.n_features + 3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

        # 상태 변수 (reset에서 초기화)
        self.current_step = 0
        self.balance = 0
        self.shares = 0
        self.total_asset = 0
        self.trades = []

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """환경을 초기 상태로 리셋한다."""
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares = 0
        self.total_asset = self.initial_balance
        self.trades = []
        self._prev_total_asset = self.initial_balance

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """행동을 실행하고 다음 상태·보상을 반환한다."""
        current_price = self.prices[self.current_step]

        # --- 테마 필터: 비활성일에는 매수 차단 ---
        theme_blocked = False
        if action == self.BUY and self._theme_active is not None:
            if not self._theme_active[self.current_step]:
                action = self.HOLD
                theme_blocked = True

        # --- 행동 실행 ---
        if action == self.BUY and self.shares == 0 and self.balance > 0:
            invest_amount = self.balance * self.trade_ratio
            buy_price = current_price * (1 + self.commission)
            self.shares = int(invest_amount // buy_price)
            if self.shares > 0:
                cost = self.shares * buy_price
                self.balance -= cost
                self.trades.append({
                    "step": self.current_step,
                    "action": "buy",
                    "price": current_price,
                    "qty": self.shares,
                })

        elif action == self.SELL and self.shares > 0:
            sell_price = current_price * (1 - self.commission)
            revenue = self.shares * sell_price
            self.balance += revenue
            self.trades.append({
                "step": self.current_step,
                "action": "sell",
                "price": current_price,
                "qty": self.shares,
            })
            self.shares = 0

        # --- 포트폴리오 가치 계산 ---
        self.total_asset = self.balance + self.shares * current_price

        # --- 보상: 포트폴리오 가치 변화율 ---
        reward = (self.total_asset - self._prev_total_asset) / self._prev_total_asset
        self._prev_total_asset = self.total_asset

        # --- 다음 스텝 ---
        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """현재 스텝의 관측값을 구성한다."""
        start = self.current_step - self.window_size
        end = self.current_step

        # 시장 데이터: (window_size, n_features)
        market_data = self.features[start:end]

        # 포지션 정보: (window_size, 3)
        current_price = self.prices[self.current_step]
        stock_value = self.shares * current_price

        if self.total_asset > 0:
            holding_ratio = stock_value / self.total_asset
            balance_ratio = self.balance / self.total_asset
        else:
            holding_ratio = 0.0
            balance_ratio = 0.0

        profit_ratio = (self.total_asset - self.initial_balance) / self.initial_balance

        position_info = np.full(
            (self.window_size, 3),
            [holding_ratio, profit_ratio, balance_ratio],
            dtype=np.float32,
        )

        obs = np.concatenate([market_data, position_info], axis=1).astype(np.float32)
        return obs

    def _get_info(self) -> dict:
        """현재 상태 정보를 반환한다."""
        return {
            "balance": self.balance,
            "shares": self.shares,
            "total_asset": self.total_asset,
            "profit": self.total_asset - self.initial_balance,
            "profit_pct": (self.total_asset - self.initial_balance) / self.initial_balance,
            "n_trades": len(self.trades),
        }

    @staticmethod
    def _build_theme_array(df: pd.DataFrame, theme_signal: dict) -> np.ndarray:
        """DataFrame 인덱스와 theme_signal을 매칭하여 bool 배열 생성."""
        active = np.ones(len(df), dtype=bool)  # 기본값: True (매매 허용)
        for i, idx in enumerate(df.index):
            # 인덱스가 날짜 형식이면 YYYYMMDD로 변환
            if hasattr(idx, "strftime"):
                date_str = idx.strftime("%Y%m%d")
            else:
                date_str = str(idx).replace("-", "")[:8]
            if date_str in theme_signal:
                active[i] = theme_signal[date_str]
        return active

    def render(self, mode="human"):
        info = self._get_info()
        print(
            f"Step {self.current_step} | "
            f"잔고: {info['balance']:,.0f} | "
            f"보유: {self.shares}주 | "
            f"총자산: {info['total_asset']:,.0f} | "
            f"수익률: {info['profit_pct']:.2%}"
        )
