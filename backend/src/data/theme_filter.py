"""테마 기반 매매 필터.

특정 테마(예: 반도체)가 시장을 주도하는 날에만 매매를 허용한다.

판단 기준:
    1. 거래대금 상위 top_n 종목을 추출
    2. 그 중 등락률 >= threshold 인 종목을 필터링
    3. 상승 종목 중 해당 테마 종목이 가장 많으면 → 테마 활성

사용법:
    filter = ThemeFilter(theme="반도체", top_n=30, threshold=5.0)
    signals = filter.build_signal(start="20180101", end="20241231")
    # signals: {date_str: True/False} 딕셔너리
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests
from bs4 import BeautifulSoup

# 네이버 금융 반도체 관련 테마 번호
THEME_IDS = {
    "반도체": [608, 12, 533, 14, 155],
}

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "theme_cache"


def fetch_theme_codes(theme: str, use_cache: bool = True) -> Set[str]:
    """네이버 금융에서 테마 종목 코드를 수집한다.

    Parameters
    ----------
    theme : str
        테마 이름. 현재 "반도체"만 지원.
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

    headers = {"User-Agent": "Mozilla/5.0"}
    codes = set()

    for tid in THEME_IDS[theme]:
        url = f"https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no={tid}"
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        for a_tag in soup.select('a[href*="main.naver?code="]'):
            code = a_tag["href"].split("code=")[1]
            codes.add(code)

    # 캐시 저장
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sorted(codes), f)

    return codes


def build_theme_signal(
    theme: str,
    start: str,
    end: str,
    top_n: int = 30,
    threshold: float = 5.0,
    use_cache: bool = True,
) -> Dict[str, bool]:
    """날짜별 테마 활성 신호를 생성한다.

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
        상승 기준 등락률(%) (기본 5.0).
    use_cache : bool
        신호 캐시 사용 여부.

    Returns
    -------
    dict[str, bool]
        {날짜문자열: 테마 활성 여부} 딕셔너리.
    """
    from pykrx import stock

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{theme}_signal_{start}_{end}_top{top_n}_thr{threshold}.json"

    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 테마 종목 코드 수집
    theme_codes = fetch_theme_codes(theme, use_cache=use_cache)

    # 모든 테마의 코드 집합 (비교용)
    all_themes = _fetch_all_theme_codes(use_cache=use_cache)

    # 날짜별 신호 계산
    signal = {}
    dates = stock.get_previous_business_days(fromdate=start, todate=end)

    for date in dates:
        date_str = date.strftime("%Y%m%d")
        try:
            # 전 종목 OHLCV (거래대금 포함)
            df = stock.get_market_ohlcv_by_ticker(date_str, market="ALL")
            if df.empty:
                signal[date_str] = False
                continue

            # 거래대금 상위 top_n
            top = df.nlargest(top_n, "거래대금")

            # 등락률 >= threshold% 인 종목
            rising = top[top["등락률"] >= threshold]
            if rising.empty:
                signal[date_str] = False
                continue

            rising_codes = set(rising.index)

            # 테마별 상승 종목 수 비교
            theme_rising_count = len(rising_codes & theme_codes)

            # 다른 테마들보다 많은지 확인
            is_dominant = True
            for other_name, other_codes in all_themes.items():
                if other_name == theme:
                    continue
                other_count = len(rising_codes & other_codes)
                if other_count > theme_rising_count:
                    is_dominant = False
                    break

            signal[date_str] = theme_rising_count > 0 and is_dominant

        except Exception:
            signal[date_str] = False

    # 캐시 저장
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(signal, f)

    return signal


def _fetch_all_theme_codes(use_cache: bool = True) -> Dict[str, Set[str]]:
    """네이버 금융 전체 테마 목록에서 각 테마별 종목 코드를 수집한다."""
    cache_path = CACHE_DIR / "all_themes.json"

    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {k: set(v) for k, v in data.items()}

    headers = {"User-Agent": "Mozilla/5.0"}
    themes = {}

    # 네이버 금융 테마 목록 페이지 (여러 페이지)
    for page in range(1, 10):
        url = f"https://finance.naver.com/sise/theme.naver?&page={page}"
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        links = soup.select('a[href*="sise_group_detail"]')
        if not links:
            break

        for link in links:
            name = link.text.strip()
            if not name:
                continue
            href = link["href"]
            tid = href.split("no=")[-1]

            detail_url = f"https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no={tid}"
            detail_res = requests.get(detail_url, headers=headers, timeout=10)
            detail_soup = BeautifulSoup(detail_res.text, "html.parser")

            codes = set()
            for a_tag in detail_soup.select('a[href*="main.naver?code="]'):
                code = a_tag["href"].split("code=")[1]
                codes.add(code)

            if codes:
                themes[name] = codes

    # 캐시 저장
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {k: sorted(v) for k, v in themes.items()}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)

    return themes
