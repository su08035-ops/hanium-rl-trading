"""여러 config를 일괄 실행하는 스윕 스크립트.

사용법:
  전체: python run_sweep.py --config-dir ../configs/
  선택: python run_sweep.py --configs dqn_dnn.yaml ppo_lstm.yaml
  학습+백테스트: python run_sweep.py --backtest
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_command(cmd, label):
    """커맨드를 실행하고 결과를 출력."""
    print(f"\n{'='*60}")
    print(f"[SWEEP] {label}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=str(SCRIPT_DIR),
        capture_output=False,
    )
    elapsed = time.time() - start
    status = "성공" if result.returncode == 0 else "실패"
    print(f"[SWEEP] {label} — {status} ({elapsed:.0f}초)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="RL Trading - 조합 스윕")
    parser.add_argument("--config-dir", type=str, default="../configs",
                        help="config 폴더 경로")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="실행할 config 파일명 목록 (없으면 base 제외 전부)")
    parser.add_argument("--backtest", action="store_true",
                        help="학습 후 백테스트도 실행")
    parser.add_argument("--backtest-only", action="store_true",
                        help="학습 건너뛰고 백테스트만 실행")
    parser.add_argument("--compare", action="store_true",
                        help="모든 실행 완료 후 비교 JSON 생성")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)

    if args.configs:
        config_files = [config_dir / c for c in args.configs]
    else:
        config_files = sorted([
            f for f in config_dir.glob("*.yaml")
            if f.name != "base.yaml"
        ])

    print(f"[SWEEP] 실행할 config: {len(config_files)}개")
    for f in config_files:
        print(f"  - {f.name}")

    total_start = time.time()
    results = {}

    for cfg_path in config_files:
        name = cfg_path.stem

        # 학습
        if not args.backtest_only:
            ok = run_command(
                ["train.py", "--config", str(cfg_path)],
                f"학습: {name}",
            )
            results[f"train_{name}"] = ok

        # 백테스트
        if args.backtest or args.backtest_only:
            ok = run_command(
                ["backtest.py", "--config", str(cfg_path), "--model", "best"],
                f"백테스트: {name}",
            )
            results[f"backtest_{name}"] = ok

    # 비교 JSON 생성
    if args.compare:
        run_command(["compare.py"], "비교 JSON 생성")

    # 요약
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"[SWEEP] 전체 완료 — {total_elapsed:.0f}초")
    print(f"{'='*60}")
    for task, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {task}")


if __name__ == "__main__":
    main()
