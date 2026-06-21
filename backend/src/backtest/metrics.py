"""성과 지표 계산."""

from typing import Dict, List

import numpy as np


def compute_metrics(equity_curve: List[float],
                    initial_balance: float) -> Dict[str, float]:
    """자산 곡선에서 주요 성과 지표를 계산한다.

    Parameters
    ----------
    equity_curve : list of float
        일별 포트폴리오 가치.
    initial_balance : float
        초기 자본금.

    Returns
    -------
    dict
        total_return, annualized_return, sharpe, mdd, win_rate 등.
    """
    # TODO: 총 수익률
    # TODO: 연환산 수익률
    # TODO: 샤프 비율 (무위험 수익률 = 0 가정)
    # TODO: 최대 낙폭 (MDD)
    # TODO: 승률 (양의 수익 거래 / 전체 거래)
    raise NotImplementedError


def compute_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """일별 수익률 배열에서 샤프 비율을 계산한다."""
    # TODO: (mean - rf) / std * sqrt(252)
    raise NotImplementedError
