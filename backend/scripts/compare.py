"""실험 결과를 모아 웹 대시보드용 통합 JSON을 생성한다.

사용법: python compare.py
       python compare.py --experiments dqn_dnn_005930 ppo_lstm_005930
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
EXP_DIR = BACKEND_DIR / "experiments"
RESULTS_DIR = BACKEND_DIR.parent / "results"


def load_experiment(exp_path: Path) -> dict:
    """실험 디렉토리에서 백테스트 결과를 로드."""
    bt_path = exp_path / "backtest_result.json"
    if not bt_path.exists():
        return None
    with open(bt_path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="RL Trading - 실험 결과 비교·통합")
    parser.add_argument("--exp-dir", type=str, default=str(EXP_DIR),
                        help="실험 결과 루트 디렉토리")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="비교할 실험명 목록 (없으면 전부)")
    parser.add_argument("--output", type=str, default=None,
                        help="통합 JSON 출력 경로")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"실험 디렉토리 없음: {exp_dir}")
        return

    # 실험 목록 결정
    if args.experiments:
        exp_names = args.experiments
    else:
        exp_names = sorted([
            d.name for d in exp_dir.iterdir()
            if d.is_dir() and (d / "backtest_result.json").exists()
        ])

    if not exp_names:
        print("백테스트 결과가 있는 실험이 없습니다. 먼저 backtest.py를 실행하세요.")
        return

    print(f"[비교] {len(exp_names)}개 실험 로드 중...")

    # 결과 수집
    experiments = []
    for name in exp_names:
        result = load_experiment(exp_dir / name)
        if result is None:
            print(f"  건너뜀: {name} (backtest_result.json 없음)")
            continue
        experiments.append(result)
        print(f"  로드: {name}")

    if not experiments:
        print("로드된 실험이 없습니다.")
        return

    # 정렬: 총 수익률 기준 내림차순
    experiments.sort(key=lambda x: x["agent"]["total_return"], reverse=True)

    # 통합 JSON 생성
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "n_experiments": len(experiments),
        "ranking": [],
        "buyhold_baseline": experiments[0].get("buyhold", {}),
        "experiments": experiments,
    }

    # 랭킹 테이블
    print(f"\n{'='*80}")
    print(f"{'순위':>4} | {'실험명':<30} | {'수익률':>8} | {'샤프':>7} | {'MDD':>8} | {'승률':>6} | {'거래':>5}")
    print(f"{'-'*80}")

    for rank, exp in enumerate(experiments, 1):
        agent = exp["agent"]
        name = exp["experiment"]
        comparison["ranking"].append({
            "rank": rank,
            "experiment": name,
            "total_return": agent["total_return"],
            "sharpe_ratio": agent["sharpe_ratio"],
            "mdd": agent["mdd"],
            "win_rate": agent["win_rate"],
            "n_trades": agent["n_trades"],
        })
        print(
            f"{rank:4d} | {name:<30} | "
            f"{agent['total_return']:>+7.2%} | "
            f"{agent['sharpe_ratio']:>7.4f} | "
            f"{agent['mdd']:>7.2%} | "
            f"{agent['win_rate']:>5.1%} | "
            f"{agent['n_trades']:>5d}"
        )

    # Buy & Hold 기준선
    bh = comparison["buyhold_baseline"]
    if bh:
        print(f"{'-'*80}")
        print(
            f"{'B&H':>4} | {'Buy & Hold (기준선)':<30} | "
            f"{bh.get('total_return', 0):>+7.2%} | "
            f"{bh.get('sharpe_ratio', 0):>7.4f} | "
            f"{bh.get('mdd', 0):>7.2%} | "
            f"{'  -':>6} | "
            f"{'1':>5}"
        )
    print(f"{'='*80}")

    # 최고 성과 요약
    best = experiments[0]
    print(f"\n최고 성과: {best['experiment']}")
    print(f"  수익률: {best['agent']['total_return']:+.2%}")
    print(f"  초과수익 (vs B&H): {best.get('excess_return', 0):+.2%}")

    # JSON 저장
    if args.output:
        output_path = Path(args.output)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / "comparison.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n통합 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
