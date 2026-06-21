"""백테스트 엔진 — 학습된 에이전트를 테스트 데이터로 평가한다."""

from typing import Dict, List

import pandas as pd

from ..agents.base_agent import BaseAgent
from ..env.trading_env import TradingEnv


def run_backtest(agent: BaseAgent, env: TradingEnv) -> Dict:
    """에이전트를 환경에서 1회 에피소드 실행하여 결과를 수집한다.

    Parameters
    ----------
    agent : BaseAgent
        학습 완료된 에이전트.
    env : TradingEnv
        테스트 기간 환경.

    Returns
    -------
    dict
        {"equity_curve": [...], "trades": [...], "final_balance": float}
    """
    # TODO: env.reset() → 매 스텝 agent.select_action(explore=False)
    # TODO: equity_curve, trades 기록
    # TODO: 결과 dict 반환
    raise NotImplementedError
