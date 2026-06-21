"""단일 실험 학습 스크립트.

사용법: python train.py --config ../configs/ppo_lstm.yaml
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RL Trading - 단일 실험 학습")
    parser.add_argument("--config", type=str, required=True,
                        help="실험 config 파일 경로")
    args = parser.parse_args()

    # TODO: 1. config 로드
    # cfg = load_config(args.config)

    # TODO: 2. 시드 고정
    # set_seed(cfg["train"]["seed"])

    # TODO: 3. 데이터 로드 + 전처리
    # df = fetch_ohlcv(cfg["ticker"], cfg["data"]["train_start"], cfg["data"]["train_end"])
    # df = add_technical_indicators(df)
    # df = normalize(df)

    # TODO: 4. 환경 생성
    # env = TradingEnv(df, cfg["env"]["initial_balance"], ...)

    # TODO: 5. 네트워크 + 에이전트 조립 (registry 사용)
    # net = NetworkRegistry.create(cfg["network"], input_dim=..., **cfg.get("net_params", {}))
    # agent = AgentRegistry.create(cfg["algo"], network=net, device=..., **cfg.get("hyperparams", {}))

    # TODO: 6. 학습 루프
    # for episode in range(cfg["train"]["episodes"]):
    #     state, _ = env.reset()
    #     done = False
    #     while not done:
    #         action = agent.select_action(state)
    #         next_state, reward, terminated, truncated, info = env.step(action)
    #         agent.train_step(...)
    #         state = next_state
    #         done = terminated or truncated

    # TODO: 7. 실험 결과 저장 (experiments/{실험명}/)

    print("TODO: 학습 로직 구현 필요")


if __name__ == "__main__":
    main()
