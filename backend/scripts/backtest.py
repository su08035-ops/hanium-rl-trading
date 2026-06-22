"""백테스트 스크립트 — 학습된 모델을 테스트 데이터로 평가한다.

사용법: python backtest.py --config ../configs/dqn_dnn.yaml --model best
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.loader import fetch_ohlcv
from src.data.preprocess import add_technical_indicators, normalize
from src.env.trading_env import TradingEnv
from src.networks.registry import NetworkRegistry
from src.agents.registry import AgentRegistry
from src.backtest.engine import run_backtest, run_buyhold_baseline
from src.backtest.metrics import compute_metrics

import importlib


def main():
    parser = argparse.ArgumentParser(description="RL Trading - 백테스트")
    parser.add_argument("--config", type=str, required=True,
                        help="실험 config 파일 경로")
    parser.add_argument("--model", type=str, default="best",
                        choices=["best", "final"],
                        help="사용할 모델 (best 또는 final)")
    args = parser.parse_args()

    # 1. config 로드
    cfg = load_config(args.config)
    algo_name = cfg["algo"]
    net_name = cfg["network"]
    exp_name = f"{algo_name}_{net_name}_{cfg['ticker']}"
    exp_dir = BACKEND_DIR / "experiments" / exp_name

    # 동적 모듈 등록
    importlib.import_module(f"src.networks.{net_name}")
    importlib.import_module(f"src.agents.{algo_name}")

    set_seed(cfg["train"]["seed"])

    # 2. 테스트 데이터 로드 + 전처리
    # 학습 + 테스트 전체를 로드한 뒤, 학습 구간 통계로 정규화
    print(f"[백테스트] 데이터 로드: {cfg['ticker']} ({cfg['data']['test_start']} ~ {cfg['data']['test_end']})")
    df_all = fetch_ohlcv(
        cfg["ticker"],
        cfg["data"]["train_start"],  # 학습 시작부터 로드 (정규화 통계용)
        cfg["data"]["test_end"],
    )
    df_all = add_technical_indicators(df_all)

    # 정규화 전 원본 종가 보존
    raw_close_all = df_all["close"].values.copy()

    # 학습 구간 통계로 정규화 (데이터 누수 방지)
    df_all = normalize(df_all, method="zscore", train_end=cfg["data"]["train_end"])

    # 테스트 구간만 추출
    test_mask = df_all.index >= cfg["data"]["test_start"]
    mask_array = np.array(test_mask)
    df_test = df_all.loc[test_mask].copy()
    raw_close_test = raw_close_all[mask_array]

    print(f"[백테스트] 테스트 데이터: {len(df_test)}일")

    # 3. 테스트 환경 생성
    env = TradingEnv(
        df=df_test,
        initial_balance=cfg["env"]["initial_balance"],
        commission=cfg["env"]["commission"],
        window_size=cfg["env"]["window_size"],
        raw_prices=raw_close_test,
    )

    # 4. 네트워크 + 에이전트 조립 + 모델 로드
    device = "cpu"
    obs_shape = env.observation_space.shape
    flat_input_dim = obs_shape[0] * obs_shape[1]

    net = NetworkRegistry.create(
        net_name,
        input_dim=flat_input_dim,
        **cfg.get("net_params", {}),
    )
    agent = AgentRegistry.create(
        algo_name,
        network=net,
        device=device,
        batch_size=cfg["train"]["batch_size"],
        num_actions=env.action_space.n,
        **cfg.get("hyperparams", {}),
    )

    model_path = exp_dir / f"{args.model}_model.pt"
    agent.load(model_path)
    print(f"[백테스트] 모델 로드: {model_path.name}")

    # 5. 백테스트 실행
    print("\n===== DQN 에이전트 백테스트 =====")
    result = run_backtest(agent, env)
    agent_metrics = compute_metrics(
        result["equity_curve"],
        cfg["env"]["initial_balance"],
        result["trades"],
    )

    print(f"  총 수익률:     {agent_metrics['total_return']:+.2%}")
    print(f"  연환산 수익률: {agent_metrics['annualized_return']:+.2%}")
    print(f"  샤프 비율:     {agent_metrics['sharpe_ratio']:.4f}")
    print(f"  최대 낙폭:     {agent_metrics['mdd']:.2%}")
    print(f"  승률:          {agent_metrics['win_rate']:.2%}")
    print(f"  거래 횟수:     {agent_metrics['n_trades']}회")
    print(f"  최종 자산:     {agent_metrics['final_balance']:,.0f}원")

    # 6. Buy & Hold 기준선
    print("\n===== Buy & Hold 기준선 =====")
    bh_env = TradingEnv(
        df=df_test,
        initial_balance=cfg["env"]["initial_balance"],
        commission=cfg["env"]["commission"],
        window_size=cfg["env"]["window_size"],
        raw_prices=raw_close_test,
    )
    bh_result = run_buyhold_baseline(bh_env)
    bh_metrics = compute_metrics(
        bh_result["equity_curve"],
        cfg["env"]["initial_balance"],
        bh_result["trades"],
    )

    print(f"  총 수익률:     {bh_metrics['total_return']:+.2%}")
    print(f"  연환산 수익률: {bh_metrics['annualized_return']:+.2%}")
    print(f"  샤프 비율:     {bh_metrics['sharpe_ratio']:.4f}")
    print(f"  최대 낙폭:     {bh_metrics['mdd']:.2%}")
    print(f"  최종 자산:     {bh_metrics['final_balance']:,.0f}원")

    # 7. 비교 요약
    diff = agent_metrics["total_return"] - bh_metrics["total_return"]
    print(f"\n===== 비교 =====")
    print(f"  DQN vs B&H 초과수익: {diff:+.2%}")

    # 8. 결과 JSON 저장
    result_data = {
        "experiment": exp_name,
        "model": args.model,
        "test_period": f"{cfg['data']['test_start']} ~ {cfg['data']['test_end']}",
        "agent": agent_metrics,
        "buyhold": bh_metrics,
        "excess_return": round(diff, 4),
    }

    result_path = exp_dir / "backtest_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {result_path}")


if __name__ == "__main__":
    main()
