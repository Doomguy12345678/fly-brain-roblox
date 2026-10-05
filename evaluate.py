"""Run a trained PPO policy in a focused Roblox game window."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from config import (
    AgentConfig,
    CaptureConfig,
    ProjectPaths,
    RegionOfInterest,
    RewardConfig,
    VisionConfig,
)
from roblox_env import RobloxAnomalyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=1_000)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    parser.add_argument("--success-template", action="append", type=Path, default=[])
    parser.add_argument("--failure-template", action="append", type=Path, default=[])
    parser.add_argument("--neutral-without-feedback", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("Episode count must be positive")
    roi = RegionOfInterest(*args.roi) if args.roi else None
    paths = ProjectPaths(root=args.root)
    environment = RobloxAnomalyEnv(
        scene_name=args.scene,
        capture_config=CaptureConfig(fps=args.fps, roi=roi),
        vision_config=VisionConfig(),
        agent_config=AgentConfig(),
        reward_config=RewardConfig(
            missing_feedback="neutral" if args.neutral_with_feedback else "error",
            success_templates=tuple(args.success_template),
            failure_templates=tuple(args.failure_template),
        ),
        paths=paths,
        max_episode_steps=args.steps_per_episode,
    )
    model = PPO.load(args.model, device="auto")
    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = environment.reset()
            total_reward = 0.0
            reports = 0
            outcomes: dict[str, int] = {}
            done = False
            while not done:
                action, _state = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = environment.step(
                    int(action)
                )
                total_reward += reward
                done = terminated or truncated
                if info["action"] == "report_anomaly" and not info["report_suppressed"]:
                    reports += 1
                    outcome = info["feedback"]
                    if outcome is not None:
                        outcomes[outcome] = outcomes.get(outcome, 0) + 1
            print(
                f"episode={episode} reward={total_reward:.3f} "
                f"reports={reports} outcomes={outcomes}"
            )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
