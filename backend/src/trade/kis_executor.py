"""한국투자증권 KIS API 실거래 연동 (mojito 기반)."""


class KISExecutor:
    """mojito를 이용한 실거래 주문 실행기.

    Parameters
    ----------
    api_key : str
        한국투자증권 앱키.
    api_secret : str
        시크릿키.
    acc_no : str
        계좌번호.
    mock : bool
        True이면 모의투자.
    """

    def __init__(self, api_key: str, api_secret: str, acc_no: str,
                 mock: bool = True):
        self.mock = mock
        # TODO: mojito.KoreaInvestment 인스턴스 생성

    def buy(self, ticker: str, qty: int, price: int = 0) -> dict:
        """매수 주문."""
        # TODO: 시장가/지정가 매수
        raise NotImplementedError

    def sell(self, ticker: str, qty: int, price: int = 0) -> dict:
        """매도 주문."""
        # TODO: 시장가/지정가 매도
        raise NotImplementedError

    def get_balance(self) -> dict:
        """잔고 조회."""
        # TODO: 보유 종목·현금 잔고 반환
        raise NotImplementedError

    def get_current_price(self, ticker: str) -> int:
        """현재가 조회."""
        # TODO: 실시간 현재가 반환
        raise NotImplementedError
