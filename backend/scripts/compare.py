"""실험 결과를 모아 웹 대시보드용 통합 JSON을 생성한다.

사용법: python compare.py --experiments exp1 exp2 --output ../../results/comparison.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RL Trading - 실험 결과 비교·통합")
    parser.add_argument("--exp-dir", type=str, default="../experiments",
                        help="실험 결과 루트 디렉토리")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="비교할 실험명 목록 (없으면 전부)")
    parser.add_argument("--output", type=str, default="../../results/comparison.json",
                        help="통합 JSON 출력 경로")
    args = parser.parse_args()

    # TODO: 1. 각 실험 폴더에서 result.json 로드
    # TODO: 2. meta 정보 통합
    # TODO: 3. results 배열에 각 실험 결과 추가
    # TODO: 4. 벤치마크 (buy & hold) 계산
    # TODO: 5. 통합 JSON 저장

    print("TODO: 비교 로직 구현 필요")


if __name__ == "__main__":
    main()
