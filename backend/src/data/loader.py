"""pykrx를 이용한 OHLCV 데이터 수집."""

import pandas as pd


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """pykrx에서 종목의 OHLCV 데이터를 가져온다.

    Parameters
    ----------
    ticker : str
        종목 코드 (예: "005930").
    start : str
        시작일 "YYYY-MM-DD".
    end : str
        종료일 "YYYY-MM-DD".

    Returns
    -------
    pd.DataFrame
        날짜 인덱스, 컬럼: open, high, low, close, volume.
    """
    # TODO: pykrx.stock.get_market_ohlcv_by_date 호출
    # TODO: 컬럼명 영문 통일
    # TODO: raw/ 폴더에 캐시 저장
    raise NotImplementedError
