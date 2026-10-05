"""Configuration models for the screen-based Roblox agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RegionOfInterest:
    """A rectangle in screen coordinates."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.left < 0 or self.top < 0:
            raise ValueError("ROI left and top must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")

    def as_mss_region(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class CaptureConfig:
    """Settings for desktop capture and observation sizing."""

    fps: float = 5.0
    roi: RegionOfInterest | None = None
    output_width: int = 160
    output_height: int = 120

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("Capture FPS must be positive")
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("Output dimensions must be positive")


@dataclass(frozen=True)
class VisionConfig:
    """Thresholds used by frame comparison and change detection."""

    difference_threshold: int = 24
    scene_change_threshold: float = 0.22
    similarity_threshold: float = 0.82
    template_threshold: float = 0.80
    history_length: int = 4
    object_min_area: int = 80

    def __post_init__(self) -> None:
        if not 0 <= self.difference_threshold <= 255:
            raise ValueError("Difference threshold must be in [0, 255]")
        for name, value in (
            ("scene change", self.scene_change_threshold),
            ("similarity", self.similarity_threshold),
            ("template", self.template_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name.title()} threshold must be in [0, 1]")
        if self.history_length < 1:
            raise ValueError("History length must be at least one")
        if self.object_min_area < 1:
            raise ValueError("Minimum object area must be positive")


@dataclass(frozen=True)
class RewardConfig:
    """Rewards and game-specific visual feedback templates.

    Templates are image files whose visible screen regions indicate a
    successful or unsuccessful report. Configure them separately for each game.
    """

    correct_report: float = 1.0
    incorrect_report: float = -1.0
    neutral_step: float = -0.001
    missing_feedback: str = "error"
    feedback_timeout_seconds: float = 1.0
    feedback_poll_interval: float = 0.1
    success_templates: tuple[Path, ...] = ()
    failure_templates: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.missing_feedback not in {"error", "neutral"}:
            raise ValueError("missing_feedback must be 'error' or 'neutral'")
        if self.feedback_timeout_seconds < 0 or self.feedback_poll_interval <= 0:
            raise ValueError("Feedback timeout must be non-negative and poll interval positive")
        if self.missing_feedback == "error" and (
            not self.success_templates or not self.failure_templates
        ):
            raise ValueError(
                "Configure success and failure templates or set "
                "missing_feedback='neutral'"
            )


@dataclass(frozen=True)
class AgentConfig:
    """Key bindings for generic game actions and PPO settings."""

    key_bindings: Mapping[str, str] = field(
        default_factory=lambda: {
            "look_left": "left",
            "look_right": "right",
            "look_up": "up",
            "look_down": "down",
            "interact": "e",
            "report_anomaly": "r",
            "open_door": "e",
            "close_door": "q",
        }
    )
    action_hold_seconds: float = 0.12
    report_cooldown_steps: int = 15
    total_timesteps: int = 100_000
    checkpoint_frequency: int = 10_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64

    def __post_init__(self) -> None:
        required_actions = {
            "look_left",
            "look_right",
            "look_up",
            "look_down",
            "interact",
            "report_anomaly",
            "open_door",
            "close_door",
        }
        missing = required_actions.difference(self.key_bindings)
        if missing:
            raise ValueError(f"Missing key bindings: {', '.join(sorted(missing))}")
        if self.action_hold_seconds < 0:
            raise ValueError("Action hold duration cannot be negative")
        if self.report_cooldown_steps < 0:
            raise ValueError("Report cooldown cannot be negative")
        if self.total_timesteps < 1 or self.checkpoint_frequency < 1:
            raise ValueError("Training and checkpoint steps must be positive")
        if self.n_steps < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("PPO settings must be positive")


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used for outputs and saved experience."""

    root: Path = Path(__file__).resolve().parent
    models: Path | None = None
    logs: Path | None = None
    screenshots: Path | None = None
    memory: Path | None = None
    replays: Path | None = None

    def __post_init__(self) -> None:
        for name in ("models", "logs", "screenshots", "memory", "replays"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, self.root / name)

    def create_directories(self) -> None:
        for path in (self.models, self.logs, self.screenshots, self.memory, self.replays):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
