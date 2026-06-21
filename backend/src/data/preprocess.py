"""데이터 전처리 — 기술지표 생성 및 정규화."""

import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 데이터프레임에 기술지표 컬럼을 추가한다.

    Parameters
    ----------
    df : pd.DataFrame
        open, high, low, close, volume 컬럼 포함.

    Returns
    -------
    pd.DataFrame
        기술지표가 추가된 데이터프레임.
    """
    # TODO: 이동평균 (MA5, MA20, MA60)
    # TODO: RSI (14일)
    # TODO: MACD
    # TODO: 볼린저 밴드
    # TODO: 거래량 이동평균
    raise NotImplementedError


def normalize(df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    """수치 컬럼을 정규화한다.

    Parameters
    ----------
    df : pd.DataFrame
    method : str
        "zscore" 또는 "minmax".

    Returns
    -------
    pd.DataFrame
    """
    # TODO: 학습 구간 통계 기반 정규화 (데이터 누수 방지)
    raise NotImplementedError
