"""백테스트 엔진 — 학습된 에이전트를 테스트 데이터로 평가한다."""

from typing import Dict, List

import numpy as np

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
        equity_curve, trades, final_info 포함.
    """
    state, info = env.reset()
    equity_curve = [info["total_asset"]]
    done = False

    while not done:
        action = agent.select_action(state, explore=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        equity_curve.append(info["total_asset"])
        state = next_state
        done = terminated or truncated

    return {
        "equity_curve": equity_curve,
        "trades": env.trades,
        "final_info": info,
    }


def run_buyhold_baseline(env: TradingEnv) -> Dict:
    """Buy & Hold 기준선: 첫 날 전량 매수 후 보유."""
    state, info = env.reset()
    initial_balance = info["total_asset"]
    prices = env.prices
    start_idx = env.window_size

    # 첫 날 전량 매수
    buy_price = prices[start_idx] * (1 + env.commission)
    shares = int(initial_balance // buy_price)
    cost = shares * buy_price
    remaining_cash = initial_balance - cost

    equity_curve = [initial_balance]
    for i in range(start_idx + 1, len(prices) - 1):
        equity_curve.append(remaining_cash + shares * prices[i])

    return {
        "equity_curve": equity_curve,
        "trades": [
            {"step": start_idx, "action": "buy", "price": prices[start_idx], "qty": shares},
        ],
        "final_info": {
            "total_asset": equity_curve[-1],
            "profit_pct": (equity_curve[-1] - initial_balance) / initial_balance,
        },
    }
