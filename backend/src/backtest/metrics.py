"""성과 지표 계산."""

from typing import Dict, List

import numpy as np


def compute_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """일별 수익률 배열에서 연환산 샤프 비율을 계산한다."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / 252
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def compute_mdd(equity_curve: np.ndarray) -> float:
    """최대 낙폭(MDD)을 계산한다. 음수로 반환."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(np.min(drawdown))


def compute_metrics(equity_curve: List[float],
                    initial_balance: float,
                    trades: List[dict] = None) -> Dict[str, float]:
    """자산 곡선에서 주요 성과 지표를 계산한다."""
    eq = np.array(equity_curve, dtype=np.float64)

    # 총 수익률
    total_return = (eq[-1] - initial_balance) / initial_balance

    # 일별 수익률
    daily_returns = np.diff(eq) / eq[:-1]

    # 연환산 수익률 (거래일 252일 기준)
    n_days = len(eq) - 1
    if n_days > 0 and eq[-1] > 0:
        annualized_return = (eq[-1] / initial_balance) ** (252 / n_days) - 1
    else:
        annualized_return = 0.0

    # 샤프 비율
    sharpe = compute_sharpe(daily_returns)

    # 최대 낙폭
    mdd = compute_mdd(eq)

    # 승률 (매도 거래 기준: 매도 시 수익 > 0인 비율)
    win_rate = 0.0
    if trades:
        sell_trades = [t for t in trades if t["action"] == "sell"]
        if sell_trades:
            # 매수-매도 쌍으로 수익률 계산
            buy_trades = [t for t in trades if t["action"] == "buy"]
            wins = 0
            for i, sell in enumerate(sell_trades):
                if i < len(buy_trades):
                    if sell["price"] > buy_trades[i]["price"]:
                        wins += 1
            win_rate = wins / len(sell_trades)

    # 총 거래 횟수
    n_trades = len(trades) if trades else 0

    return {
        "total_return": round(total_return, 4),
        "annualized_return": round(annualized_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "mdd": round(mdd, 4),
        "win_rate": round(win_rate, 4),
        "n_trades": n_trades,
        "final_balance": round(eq[-1], 0),
        "initial_balance": initial_balance,
    }
