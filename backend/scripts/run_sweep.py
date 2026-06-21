"""여러 config를 일괄 실행하는 스윕 스크립트.

사용법: python run_sweep.py --config-dir ../configs/
        또는 특정 config 지정: python run_sweep.py --configs dqn_dnn.yaml ppo_lstm.yaml
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RL Trading - 조합 스윕")
    parser.add_argument("--config-dir", type=str, default="../configs",
                        help="config 폴더 경로")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="실행할 config 파일명 목록 (없으면 base 제외 전부)")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)

    if args.configs:
        config_files = [config_dir / c for c in args.configs]
    else:
        config_files = [
            f for f in config_dir.glob("*.yaml")
            if f.name != "base.yaml"
        ]

    # TODO: 각 config에 대해 train.py를 순차 또는 병렬 실행
    for cfg_path in sorted(config_files):
        print(f"[SWEEP] 실행: {cfg_path.name}")
        # subprocess.run(["python", "train.py", "--config", str(cfg_path)])

    print("TODO: 스윕 로직 구현 필요")


if __name__ == "__main__":
    main()
