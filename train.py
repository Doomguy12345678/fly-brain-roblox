"""Train a PPO policy against a focused, running Roblox game window."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from config import (
    AgentConfig,
    CaptureConfig,
    ProjectPaths,
    RegionOfInterest,
    RewardConfig,
    VisionConfig,
)
from feature_extractor import ScreenFeatureExtractor
from roblox_env import RobloxAnomalyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="Name of the room/reference baseline")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    parser.add_argument("--success-template", action="append", type=Path, default=[])
    parser.add_argument("--failure-template", action="append", type=Path, default=[])
    parser.add_argument("--neutral-without-feedback", action="store_true")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roi = RegionOfInterest(*args.roi) if args.roi else None
    capture_config = CaptureConfig(fps=args.fps, roi=roi)
    vision_config = VisionConfig()
    agent_config = AgentConfig(total_timesteps=args.timesteps)
    paths = ProjectPaths(root=args.root)
    paths.create_directories()

    reward_config = RewardConfig(
        missing_feedback="neutral" if args.neutral_without_feedback else "error",
        success_templates=tuple(args.success_template),
        failure_templates=tuple(args.failure_template),
    )
    environment = RobloxAnomalyEnv(
        scene_name=args.scene,
        capture_config=capture_config,
        vision_config=vision_config,
        agent_config=agent_config,
        reward_config=reward_config,
        paths=paths,
    )
    monitored_environment = Monitor(environment, filename=str(paths.logs / args.scene))
    checkpoint_callback = CheckpointCallback(
        save_freq=agent_config.checkpoint_frequency,
        save_path=str(paths.models),
        name_prefix=f"{args.scene}_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=monitored_environment,
        learning_rate=agent_config.learning_rate,
        n_steps=agent_config.n_steps,
        batch_size=agent_config.batch_size,
        tensorboard_log=str(paths.logs / "tensorboard"),
        seed=args.seed,
        verbose=1,
        policy_kwargs={
            "features_extractor_class": ScreenFeatureExtractor,
            "normalize_images": False,
        },
    )
    try:
        model.learn(
            total_timesteps=agent_config.total_timesteps,
            callback=checkpoint_callback,
            tb_log_name=f"{args.scene}_ppo",
        )
        model.save(paths.models / f"{args.scene}_ppo_final")
    finally:
        monitored_environment.close()


if __name__ == "__main__":
    main()
