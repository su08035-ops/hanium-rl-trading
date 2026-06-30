"""단일 실험 학습 스크립트.

사용법:
  python train.py --config ../configs/dqn_dnn.yaml
  python train.py --config ../configs/dqn_dnn_2023.yaml --resume experiments/dqn_dnn_005930/best_model.pt
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

# backend/ 를 모듈 경로에 추가
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger, log_metrics
from src.data.loader import fetch_ohlcv
from src.data.preprocess import add_technical_indicators, normalize
from src.env.trading_env import TradingEnv
from src.data.theme_filter import build_theme_signal
from src.networks.registry import NetworkRegistry
from src.agents.registry import AgentRegistry


def register_modules(algo_name: str, net_name: str):
    """config에 명시된 알고리즘/네트워크 모듈을 동적 import하여 레지스트리에 등록."""
    importlib.import_module(f"src.networks.{net_name}")
    importlib.import_module(f"src.agents.{algo_name}")


def get_device(cfg_device: str) -> str:
    if cfg_device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def main():
    parser = argparse.ArgumentParser(description="RL Trading - 단일 실험 학습")
    parser.add_argument("--config", type=str, required=True,
                        help="실험 config 파일 경로")
    parser.add_argument("--resume", type=str, default=None,
                        help="이어서 학습할 체크포인트 경로 (.pt)")
    args = parser.parse_args()

    # 1. config 로드
    cfg = load_config(args.config)
    algo_name = cfg["algo"]
    net_name = cfg["network"]
    exp_name = f"{algo_name}_{net_name}_{cfg['ticker']}"

    # 모듈 동적 등록
    register_modules(algo_name, net_name)

    # 실험 결과 디렉토리
    exp_dir = BACKEND_DIR / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 2. 로거 + 시드
    logger = setup_logger(exp_name, exp_dir)
    set_seed(cfg["train"]["seed"])
    logger.info(f"실험 시작: {exp_name}")
    logger.info(f"Config: {cfg}")

    # 3. 데이터 로드 + 전처리
    logger.info("데이터 로드 중...")
    df = fetch_ohlcv(
        cfg["ticker"],
        cfg["data"]["train_start"],
        cfg["data"]["train_end"],
    )
    df = add_technical_indicators(df)

    # 정규화 전 원본 종가 보존 (매매 가격 계산용)
    raw_close = df["close"].values.copy()

    df = normalize(df, method="zscore", train_end=cfg["data"]["train_end"])
    logger.info(f"데이터 shape: {df.shape}")

    # 3-1. 테마 필터 (config에 theme 설정이 있으면 활성화)
    theme_signal = None
    theme_cfg = cfg.get("theme_filter")
    if theme_cfg and theme_cfg.get("enabled", False):
        logger.info(f"테마 필터 활성화: {theme_cfg['name']} (top_n={theme_cfg.get('top_n', 30)}, threshold={theme_cfg.get('threshold', 5.0)}%)")
        theme_signal = build_theme_signal(
            theme=theme_cfg["name"],
            start=cfg["data"]["train_start"],
            end=cfg["data"]["train_end"],
            top_n=theme_cfg.get("top_n", 30),
            threshold=theme_cfg.get("threshold", 5.0),
        )
        active_days = sum(1 for v in theme_signal.values() if v)
        total_days = len(theme_signal)
        logger.info(f"  - 테마 활성일: {active_days}/{total_days}일 ({active_days/max(total_days,1)*100:.1f}%)")

    # 4. 환경 생성
    env = TradingEnv(
        df=df,
        initial_balance=cfg["env"]["initial_balance"],
        commission=cfg["env"]["commission"],
        window_size=cfg["env"]["window_size"],
        raw_prices=raw_close,
        theme_signal=theme_signal,
    )

    # 5. 네트워크 + 에이전트 조립
    device = get_device(cfg["train"]["device"])
    obs_shape = env.observation_space.shape  # (window_size, n_features+3)
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
    logger.info(f"네트워크: {net_name} (input={flat_input_dim}, output={net.output_dim})")
    logger.info(f"에이전트: {algo_name} (device={device})")

    # 5-1. 체크포인트 로드 (fine-tuning)
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = BACKEND_DIR / resume_path
        agent.load(resume_path)
        logger.info(f"체크포인트 로드: {resume_path}")
        epsilon = getattr(agent, "epsilon", None)
        if epsilon is not None:
            logger.info(f"  - epsilon 복원: {epsilon:.4f}")

    # 6. 학습 루프
    episodes = cfg["train"]["episodes"]
    log_interval = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]
    metrics_path = exp_dir / "metrics.jsonl"

    best_reward = float("-inf")
    start_time = time.time()

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0
        loss_count = 0

        done = False
        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, terminated)
            result = agent.train_step()

            if not result.get("skipped", False):
                episode_loss += result["loss"]
                loss_count += 1

            episode_reward += reward
            steps += 1
            state = next_state
            done = terminated or truncated

        # 에피소드 종료 콜백 (epsilon 감소, 타깃 갱신 등)
        agent.on_episode_end(ep)

        # 메트릭 기록
        avg_loss = episode_loss / max(loss_count, 1)
        epsilon = getattr(agent, "epsilon", 0.0)
        metrics = {
            "reward": round(episode_reward, 6),
            "avg_loss": round(avg_loss, 6),
            "epsilon": round(epsilon, 4),
            "steps": steps,
            "total_asset": round(info["total_asset"], 0),
            "profit_pct": round(info["profit_pct"], 4),
            "n_trades": info["n_trades"],
        }
        log_metrics(metrics_path, ep, metrics)

        # 로그 출력
        if ep % log_interval == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"EP {ep:4d}/{episodes} | "
                f"보상: {episode_reward:+.4f} | "
                f"손실: {avg_loss:.6f} | "
                f"ε: {epsilon:.3f} | "
                f"자산: {info['total_asset']:,.0f} | "
                f"수익: {info['profit_pct']:+.2%} | "
                f"거래: {info['n_trades']}회 | "
                f"경과: {elapsed:.0f}s"
            )

        # 베스트 모델 저장
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(exp_dir / "best_model.pt")

        # 정기 체크포인트
        if ep % save_interval == 0:
            agent.save(exp_dir / f"checkpoint_ep{ep}.pt")

    # 7. 최종 모델 저장
    agent.save(exp_dir / "final_model.pt")
    elapsed = time.time() - start_time
    logger.info(f"학습 완료! 총 {elapsed:.0f}초 소요")
    logger.info(f"최고 에피소드 보상: {best_reward:.6f}")
    logger.info(f"결과 저장: {exp_dir}")


if __name__ == "__main__":
    main()
