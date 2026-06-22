"""pykrx를 이용한 OHLCV 데이터 수집."""

from pathlib import Path

import pandas as pd
from pykrx import stock


RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_ohlcv(ticker: str, start: str, end: str,
                use_cache: bool = True) -> pd.DataFrame:
    """pykrx에서 종목의 OHLCV 데이터를 가져온다.

    Parameters
    ----------
    ticker : str
        종목 코드 (예: "005930").
    start : str
        시작일 "YYYY-MM-DD".
    end : str
        종료일 "YYYY-MM-DD".
    use_cache : bool
        True이면 이전에 저장한 CSV를 재사용.

    Returns
    -------
    pd.DataFrame
        날짜 인덱스, 컬럼: open, high, low, close, volume.
    """
    # 날짜 형식 변환: "YYYY-MM-DD" → "YYYYMMDD"
    start_fmt = start.replace("-", "")
    end_fmt = end.replace("-", "")

    cache_path = RAW_DIR / f"{ticker}_{start_fmt}_{end_fmt}.csv"

    # 캐시 확인
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"[loader] 캐시 로드: {cache_path.name}")
        return df

    # pykrx에서 데이터 수집
    print(f"[loader] pykrx 수집: {ticker} ({start} ~ {end})")
    df = stock.get_market_ohlcv_by_date(start_fmt, end_fmt, ticker)

    if df.empty:
        raise ValueError(f"데이터 없음: {ticker} ({start} ~ {end})")

    # 컬럼명 영문 통일
    df.columns = ["open", "high", "low", "close", "volume", "change"]
    df.index.name = "date"

    # 등락률 컬럼 제거 (전처리에서 직접 계산)
    df = df.drop(columns=["change"])

    # 캐시 저장
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"[loader] 캐시 저장: {cache_path.name}")

    return df


def get_ticker_name(ticker: str) -> str:
    """티커 코드에서 종목명을 반환한다."""
    return stock.get_market_ticker_name(ticker)
