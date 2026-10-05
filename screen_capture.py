"""Real-time desktop capture for screen-only game observations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from types import TracebackType

import cv2
import mss
import numpy as np
from numpy.typing import NDArray

from config import CaptureConfig


Frame = NDArray[np.uint8]


class ScreenCapture:
    """Capture a configured screen region and resize it for the agent."""

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._capture: mss.mss | None = None

    def __enter__(self) -> ScreenCapture:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def start(self) -> None:
        if self._capture is None:
            self._capture = mss.mss()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None

    def capture(self) -> Frame:
        """Return one BGR frame resized to the configured dimensions."""
        if self._capture is None:
            self.start()
        if self._capture is None:
            raise RuntimeError("Screen capture failed to initialize")

        monitor = (
            self.config.roi.as_mss_region()
            if self.config.roi is not None
            else self._capture.monitors[1]
        )
        screenshot = np.asarray(self._capture.grab(monitor), dtype=np.uint8)
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        return cv2.resize(
            frame,
            (self.config.output_width, self.config.output_height),
            interpolation=cv2.INTER_AREA,
        )

    def frames(self) -> Iterator[Frame]:
        """Yield frames at approximately the configured capture rate."""
        interval = 1.0 / self.config.fps
        deadline = time.monotonic()
        while True:
            yield self.capture()
            deadline += interval
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()
