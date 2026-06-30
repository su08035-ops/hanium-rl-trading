"""테마 기반 매매 필터.

특정 테마(예: 반도체)가 시장을 주도하는 날에만 매매를 허용한다.

판단 기준:
    1. 시총 상위 종목들의 일별 OHLCV를 수집 (pykrx)
    2. 매일 거래대금 상위 top_n 종목 추출
    3. 그 중 등락률 >= threshold% 인 종목을 필터링
    4. 상승 종목 중 해당 테마 종목이 1개 이상이고 가장 많으면 → 테마 활성

사용법:
    signal = build_theme_signal("반도체", "20180101", "20241231")
    # signal: {"20180102": True, "20180103": False, ...}
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# 네이버 금융 반도체 관련 테마 번호
THEME_IDS = {
    "반도체": [608, 12, 533, 14, 155],
}

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "theme_cache"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_theme_codes(theme: str, use_cache: bool = True) -> Set[str]:
    """네이버 금융에서 테마 종목 코드를 수집한다.

    Parameters
    ----------
    theme : str
        테마 이름. 현재 "반도체" 지원.
    use_cache : bool
        True이면 캐시 파일이 있으면 재사용.

    Returns
    -------
    set[str]
        6자리 종목코드 집합.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{theme}_codes.json"

    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return set(json.load(f))

    if theme not in THEME_IDS:
        raise ValueError(f"지원하지 않는 테마: {theme}. 가능: {list(THEME_IDS.keys())}")

    codes = set()
    for tid in THEME_IDS[theme]:
        url = f"https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no={tid}"
        res = requests.get(url, headers=_HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        for a_tag in soup.select('a[href*="main.naver?code="]'):
            code = a_tag["href"].split("code=")[1]
            codes.add(code)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sorted(codes), f)

    return codes


def _fetch_market_cap_codes(n_pages: int = 6) -> List[str]:
    """네이버 금융에서 시가총액 상위 종목 코드를 수집한다.

    KOSPI + KOSDAQ 각 n_pages(기본6) × 50종목 = 최대 600종목.
    거래대금 상위 30개는 대부분 시총 상위에 포함된다.
    """
    cache_path = CACHE_DIR / "market_cap_codes.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 시총 순위는 자주 바뀌지 않으므로 캐시 사용
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    codes = []
    seen = set()
    for sosok in [0, 1]:  # 0=KOSPI, 1=KOSDAQ
        for page in range(1, n_pages + 1):
            url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
            res = requests.get(url, headers=_HEADERS, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            for a_tag in soup.select('a[href*="main.naver?code="]'):
                code = a_tag["href"].split("code=")[1]
                if code not in seen:
                    codes.append(code)
                    seen.add(code)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(codes, f)

    return codes


def _fetch_all_theme_codes(use_cache: bool = True) -> Dict[str, Set[str]]:
    """네이버 금융 전체 테마 목록에서 각 테마별 종목 코드를 수집한다."""
    cache_path = CACHE_DIR / "all_themes.json"

    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {k: set(v) for k, v in data.items()}

    themes = {}
    for page in range(1, 10):
        url = f"https://finance.naver.com/sise/theme.naver?&page={page}"
        res = requests.get(url, headers=_HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        links = soup.select('a[href*="sise_group_detail"]')
        if not links:
            break

        for link in links:
            name = link.text.strip()
            if not name:
                continue
            tid = link["href"].split("no=")[-1]

            detail_url = f"https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no={tid}"
            detail_res = requests.get(detail_url, headers=_HEADERS, timeout=10)
            detail_soup = BeautifulSoup(detail_res.text, "html.parser")

            codes = set()
            for a_tag in detail_soup.select('a[href*="main.naver?code="]'):
                code = a_tag["href"].split("code=")[1]
                codes.add(code)

            if codes:
                themes[name] = codes
            time.sleep(0.2)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {k: sorted(v) for k, v in themes.items()}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)

    return themes


def build_theme_signal(
    theme: str,
    start: str,
    end: str,
    top_n: int = 30,
    threshold: float = 3.0,
    min_theme_count: int = 1,
    use_cache: bool = True,
) -> Dict[str, bool]:
    """날짜별 테마 활성 신호를 생성한다.

    판단 기준:
        거래대금 상위 top_n 중 등락률 >= threshold% 인 종목에서
        해당 테마 종목이 min_theme_count개 이상이면 활성.

    Parameters
    ----------
    theme : str
        테마 이름 (예: "반도체").
    start : str
        시작일 (YYYYMMDD).
    end : str
        종료일 (YYYYMMDD).
    top_n : int
        거래대금 상위 종목 수 (기본 30).
    threshold : float
        상승 기준 등락률(%) (기본 3.0).
    min_theme_count : int
        활성 판정에 필요한 최소 테마 종목 수 (기본 1).
    use_cache : bool
        신호 캐시 사용 여부.

    Returns
    -------
    dict[str, bool]
        {날짜문자열: 테마 활성 여부} 딕셔너리.
    """
    import pandas as pd
    from pykrx import stock

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = (
        CACHE_DIR
        / f"{theme}_signal_{start}_{end}_top{top_n}_thr{threshold}_min{min_theme_count}.json"
    )

    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 1. 테마 종목 수집
    theme_codes = fetch_theme_codes(theme, use_cache=use_cache)

    # 2. 시총 상위 종목 코드 수집
    market_codes = _fetch_market_cap_codes()
    logger.info(f"시총 상위 {len(market_codes)}개 종목의 OHLCV 수집 시작...")

    # 3. 각 종목의 기간별 OHLCV 수집 (pykrx 개별 종목 조회)
    frames = []
    for i, code in enumerate(market_codes):
        try:
            df = stock.get_market_ohlcv_by_date(start, end, code)
            if df.empty:
                continue
            df = df.copy()
            df["code"] = code
            df["거래대금"] = df["종가"] * df["거래량"]
            frames.append(df[["code", "거래대금", "등락률"]])
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            logger.info(f"  수집 진행: {i + 1}/{len(market_codes)}")
        time.sleep(0.1)

    if not frames:
        logger.warning("OHLCV 수집 실패. 전 거래일 비활성 처리.")
        return {}

    all_df = pd.concat(frames)
    logger.info(f"OHLCV 수집 완료. 총 {len(all_df)}행")

    # 4. 날짜별 테마 활성 판단
    signal = {}
    for date, group in all_df.groupby(all_df.index):
        date_str = date.strftime("%Y%m%d")

        # 거래대금 상위 top_n
        top = group.nlargest(top_n, "거래대금")

        # 등락률 >= threshold%
        rising = top[top["등락률"] >= threshold]
        if rising.empty:
            signal[date_str] = False
            continue

        rising_codes = set(rising["code"])
        theme_count = len(rising_codes & theme_codes)

        # 테마 종목이 min_theme_count개 이상이면 활성
        signal[date_str] = theme_count >= min_theme_count

    # 캐시 저장
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(signal, f)

    active_days = sum(1 for v in signal.values() if v)
    logger.info(f"테마 신호 생성 완료: {active_days}/{len(signal)}일 활성")

    return signal
