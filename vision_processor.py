"""Convert captured frames into policy-ready image observations."""

from __future__ import annotations

from collections import deque
from typing import TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

from anomaly_detector import AnomalyDetector, DetectionResult
from config import VisionConfig


ImageArray = NDArray[np.uint8]


class VisionObservation(TypedDict):
    frame: ImageArray
    difference: ImageArray
    history: ImageArray


class VisionProcessor:
    """Maintain a short frame history and produce image/difference channels."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.detector = AnomalyDetector(config)
        self._history: deque[ImageArray] = deque(maxlen=config.history_length)

    def reset_history(self) -> None:
        self._history.clear()

    def process(
        self,
        frame: ImageArray,
        reference: ImageArray,
    ) -> tuple[VisionObservation, DetectionResult]:
        """Return RGB frame, difference map, and historical RGB frames."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR frame")

        result = self.detector.compare(frame, reference)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._history.append(rgb_frame.copy())
        while len(self._history) < self.config.history_length:
            self._history.appendleft(rgb_frame.copy())

        observation: VisionObservation = {
            "frame": rgb_frame,
            "difference": result.difference_map[..., np.newaxis].copy(),
            "history": np.concatenate(tuple(self._history), axis=2),
        }
        return observation, result
