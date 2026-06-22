"""데이터 전처리 — 기술지표 생성 및 정규화."""

import numpy as np
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
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- 이동평균 ---
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()

    # --- RSI (14일) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- 볼린저 밴드 (20일) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid

    # --- 거래량 이동평균 ---
    df["vol_ma5"] = volume.rolling(5).mean()
    df["vol_ma20"] = volume.rolling(20).mean()

    # --- 수익률 ---
    df["daily_return"] = close.pct_change()

    # --- ATR (Average True Range, 14일) ---
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # NaN 행 제거 (이동평균 초기 구간)
    df = df.dropna().copy()

    return df


def normalize(df: pd.DataFrame, method: str = "zscore",
              train_end: str = None) -> pd.DataFrame:
    """수치 컬럼을 정규화한다.

    Parameters
    ----------
    df : pd.DataFrame
    method : str
        "zscore" 또는 "minmax".
    train_end : str
        학습 구간 종료일. 이 날짜까지의 통계로 정규화하여 데이터 누수를 방지.
        None이면 전체 데이터 사용.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if train_end:
        train_data = df.loc[:train_end, numeric_cols]
    else:
        train_data = df[numeric_cols]

    if method == "zscore":
        mean = train_data.mean()
        std = train_data.std().replace(0, 1)
        df[numeric_cols] = (df[numeric_cols] - mean) / std
    elif method == "minmax":
        min_val = train_data.min()
        max_val = train_data.max()
        range_val = (max_val - min_val).replace(0, 1)
        df[numeric_cols] = (df[numeric_cols] - min_val) / range_val
    else:
        raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'minmax'.")

    return df
