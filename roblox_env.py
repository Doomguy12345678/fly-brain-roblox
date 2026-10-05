"""Gymnasium environment for interacting with a live Roblox game window."""

from __future__ import annotations

import json
import time
from enum import IntEnum
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from anomaly_detector import DetectionResult
from config import AgentConfig, CaptureConfig, ProjectPaths, RewardConfig, VisionConfig
from memory_manager import MemoryManager
from screen_capture import ScreenCapture
from vision_processor import VisionObservation, VisionProcessor


class GameAction(IntEnum):
    LOOK_LEFT = 0
    LOOK_RIGHT = 1
    LOOK_UP = 2
    LOOK_DOWN = 3
    INTERACT = 4
    REPORT_ANOMALY = 5
    OPEN_DOOR = 6
    CLOSE_DOOR = 7
    WAIT = 8


class RobloxAnomalyEnv(gym.Env[VisionObservation, int]):
    """Control a Roblox game using only captured screen pixels and key inputs.

    Before reset, focus Roblox and either save a normal-room reference under
    ``paths.memory/references/<scene_name>.png`` or let the first reset capture it.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    _ACTION_NAMES = {
        GameAction.LOOK_LEFT: "look_left",
        GameAction.LOOK_RIGHT: "look_right",
        GameAction.LOOK_UP: "look_up",
        GameAction.LOOK_DOWN: "look_down",
        GameAction.INTERACT: "interact",
        GameAction.REPORT_ANOMALY: "report_anomaly",
        GameAction.OPEN_DOOR: "open_door",
        GameAction.CLOSE_DOOR: "close_door",
    }

    def __init__(
        self,
        scene_name: str,
        capture_config: CaptureConfig,
        vision_config: VisionConfig,
        agent_config: AgentConfig,
        reward_config: RewardConfig,
        paths: ProjectPaths,
        max_episode_steps: int = 1_000,
        archive_every_n_steps: int = 10,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if max_episode_steps < 1 or archive_every_n_steps < 1:
            raise ValueError("Episode and archive intervals must be positive")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.scene_name = scene_name
        self.capture = ScreenCapture(capture_config)
        self.vision = VisionProcessor(vision_config)
        self.agent_config = agent_config
        self.reward_config = reward_config
        self.paths = paths
        self.max_episode_steps = max_episode_steps
        self.archive_every_n_steps = archive_every_n_steps
        self.render_mode = render_mode
        self.memory = MemoryManager(paths.memory)
        self._step_count = 0
        self._steps_since_report = agent_config.report_cooldown_steps
        self._reference: NDArray[np.uint8] | None = None
        self._last_frame: NDArray[np.uint8] | None = None
        self._last_observation: VisionObservation | None = None
        self._feedback_templates = self._load_feedback_templates()

        height, width = capture_config.output_height, capture_config.output_width
        self.observation_space = spaces.Dict(
            {
                "frame": spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8),
                "difference": spaces.Box(0, 255, shape=(height, width, 1), dtype=np.uint8),
                "history": spaces.Box(
                    0,
                    255,
                    shape=(height, width, vision_config.history_length * 3),
                    dtype=np.uint8,
                ),
            }
        )
        self.action_space = spaces.Discrete(len(GameAction))
        self.paths.create_directories()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VisionObservation, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._steps_since_report = self.agent_config.report_cooldown_steps
        self.vision.reset_history()

        reference_path = self.memory.reference_path(self.scene_name)
        if reference_path.exists():
            self._reference = self.memory.load_reference(self.scene_name)
        else:
            self._reference = self.capture.capture_at_rate()
            self.memory.save_reference(self.scene_name, self._reference)

        frame = self.capture.capture_at_rate()
        observation, result = self.vision.process(frame, self._reference)
        self._last_frame = frame
        self._last_observation = observation
        self.memory.record_observation(self.scene_name, result)
        return observation, {"similarity": result.similarity, "events": result.events}

    def step(
        self,
        action: int,
    ) -> tuple[VisionObservation, float, bool, bool, dict[str, Any]]:
        if self._reference is None:
            raise RuntimeError("Call reset() before step()")
        try:
            selected_action = GameAction(action)
        except ValueError as error:
            raise ValueError(f"Unknown action: {action}") from error

        action_name = self._ACTION_NAMES.get(selected_action)
        report_suppressed = (
            selected_action == GameAction.REPORT_ANOMALY
            and self._steps_since_report < self.agent_config.report_cooldown_steps
        )
        if action_name is not None and not report_suppressed:
            self._press_action(action_name)
        else:
            time.sleep(self.agent_config.action_hold_seconds)

        frame = self.capture.capture_at_rate()
        reward = self.reward_config.neutral_step
        feedback: str | None = None
        if selected_action == GameAction.REPORT_ANOMALY and not report_suppressed:
            reward, feedback, frame = self._score_report(frame)

        if selected_action == GameAction.REPORT_ANOMALY:
            if not report_suppressed:
                self._steps_since_report = 0
        else:
            self._steps_since_report += 1

        observation, result = self.vision.process(frame, self._reference)
        self._step_count += 1
        frame_path = self._archive_frame(frame) if (
            self._step_count % self.archive_every_n_steps == 0
        ) else None
        self.memory.record_observation(self.scene_name, result, frame_path)

        truncated = self._step_count >= self.max_episode_steps
        info: dict[str, Any] = {
            "action": selected_action.name.lower(),
            "similarity": result.similarity,
            "events": result.events,
            "report_suppressed": report_suppressed,
            "feedback": feedback,
            "frame_path": str(frame_path) if frame_path is not None else None,
        }
        self._record_replay(action_name or "wait", reward, feedback, result, frame_path)
        self._last_frame = frame
        self._last_observation = observation
        return observation, reward, False, truncated, info

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode != "rgb_array" or self._last_frame is None:
            return None
        return cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.capture.close()

    def _press_action(self, action_name: str) -> None:
        import pyautogui

        key = self.agent_config.key_bindings[action_name]
        pyautogui.keyDown(key)
        try:
            time.sleep(self.agent_config.action_hold_seconds)
        finally:
            pyautogui.keyUp(key)

    def _load_feedback_templates(
        self,
    ) -> dict[str, tuple[NDArray[np.uint8], ...]]:
        templates: dict[str, tuple[NDArray[np.uint8], ...]] = {}
        for outcome, paths in (
            ("success", self.reward_config.success_templates),
            ("failure", self.reward_config.failure_templates),
        ):
            loaded: list[NDArray[np.uint8]] = []
            for path in paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Could not load {outcome} template: {path}")
                loaded.append(image)
            templates[outcome] = tuple(loaded)
        return templates

    def _score_report(
        self,
        initial_frame: NDArray[np.uint8],
    ) -> tuple[float, str | None, NDArray[np.uint8]]:
        deadline = time.monotonic() + self.reward_config.feedback_timeout_seconds
        frame = initial_frame
        while True:
            outcome = self._recognize_feedback(frame)
            if outcome is not None:
                reward = (
                    self.reward_config.correct_report
                    if outcome == "success"
                    else self.reward_config.incorrect_report
                )
                return reward, outcome, frame
            if time.monotonic() >= deadline:
                break
            time.sleep(self.reward_config.feedback_poll_interval)
            frame = self.capture.capture_at_rate()

        if self.reward_config.missing_feedback == "error":
            raise RuntimeError(
                "No report outcome matched the configured feedback templates. "
                "Capture templates from this game's success and failure screens, "
                "or explicitly set missing_feedback='neutral'."
            )
        return self.reward_config.neutral_step, "unrecognized", frame

    def _recognize_feedback(self, frame: NDArray[np.uint8]) -> str | None:
        scores: dict[str, float] = {}
        for outcome, templates in self._feedback_templates.items():
            scores[outcome] = max(
                (
                    self.vision.detector.match_template(frame, template)[0]
                    for template in templates
                    if template.shape[0] <= frame.shape[0]
                    and template.shape[1] <= frame.shape[1]
                ),
                default=0.0,
            )

        threshold = self.vision.config.template_threshold
        matches = {
            outcome: score for outcome, score in scores.items() if score >= threshold
        }
        if not matches:
            return None
        ranked = sorted(matches.items(), key=lambda item: item[1], reverse=True)
        if (
            len(ranked) > 1
            and ranked[0][1] - ranked[1][1] < self.reward_config.feedback_margin
        ):
            return None
        return ranked[0][0]

    def _archive_frame(self, frame: NDArray[np.uint8]) -> Path:
        path = self.paths.screenshots / (
            f"{self.scene_name}_{time.time_ns()}_{self._step_count:08d}.png"
        )
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Could not archive screenshot: {path}")
        return path

    def _record_replay(
        self,
        action: str,
        reward: float,
        feedback: str | None,
        result: DetectionResult,
        frame_path: Path | None,
    ) -> None:
        path = self.paths.replays / f"{self.scene_name}.jsonl"
        record = {
            "timestamp_ns": time.time_ns(),
            "step": self._step_count,
            "action": action,
            "reward": reward,
            "feedback": feedback,
            "similarity": result.similarity,
            "events": [event.kind.value for event in result.events],
            "frame_path": str(frame_path) if frame_path is not None else None,
        }
        with path.open("a", encoding="utf-8") as replay_file:
            replay_file.write(json.dumps(record) + "\n")
