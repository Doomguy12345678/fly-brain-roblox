"""Persistent normal-room references and compact observation history."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from anomaly_detector import DetectionResult


Frame = NDArray[np.uint8]


class MemoryManager:
    """Store baseline room images and append-only observation metadata."""

    def __init__(self, directory: Path, history_limit: int = 1_000) -> None:
        if history_limit < 1:
            raise ValueError("History limit must be at least one")
        self.directory = directory
        self.reference_directory = directory / "references"
        self.observation_log = directory / "observations.jsonl"
        self.history_limit = history_limit
        self._recent_observations: list[dict[str, object]] = []
        self.reference_directory.mkdir(parents=True, exist_ok=True)

    def save_reference(self, scene_name: str, frame: Frame) -> Path:
        """Save a normal-room BGR image and return its path."""
        self._validate_frame(frame)
        path = self.reference_path(scene_name)
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Could not write reference image: {path}")
        return path

    def load_reference(self, scene_name: str) -> Frame:
        """Load a saved normal-room BGR image."""
        path = self.reference_path(scene_name)
        reference = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if reference is None:
            raise FileNotFoundError(f"No reference image found for scene {scene_name!r}: {path}")
        return reference

    def reference_path(self, scene_name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", scene_name):
            raise ValueError("Scene names may contain only letters, numbers, '_' and '-'")
        return self.reference_directory / f"{scene_name}.png"

    def record_observation(
        self,
        scene_name: str,
        result: DetectionResult,
        frame_path: Path | None = None,
    ) -> dict[str, object]:
        """Append an observation summary and retain a bounded recent history."""
        record: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "scene": scene_name,
            "similarity": result.similarity,
            "scene_changed": result.scene_changed,
            "events": [
                {
                    "kind": event.kind.value,
                    "confidence": event.confidence,
                    "current_box": event.current_box,
                    "reference_box": event.reference_box,
                }
                for event in result.events
            ],
            "frame_path": str(frame_path) if frame_path is not None else None,
        }
        self._recent_observations.append(record)
        if len(self._recent_observations) > self.history_limit:
            del self._recent_observations[: len(self._recent_observations) - self.history_limit]

        self.directory.mkdir(parents=True, exist_ok=True)
        with self.observation_log.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")
        return record

    @property
    def recent_observations(self) -> tuple[dict[str, object], ...]:
        return tuple(self._recent_observations)

    @staticmethod
    def _validate_frame(frame: Frame) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("Expected a non-empty three-channel BGR frame")
