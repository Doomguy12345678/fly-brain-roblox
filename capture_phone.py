"""Observe a desktop-mirrored phone screen without sending input to the phone."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from anomaly_detector import AnomalyDetector
from config import CaptureConfig, ProjectPaths, RegionOfInterest, VisionConfig
from memory_manager import MemoryManager
from screen_capture import Frame, ScreenCapture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="Name for this room/reference")
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Desktop coordinates of the phone-mirroring window",
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    parser.add_argument(
        "--archive-every",
        type=int,
        default=5,
        help="Save every Nth captured frame (default: 5)",
    )
    parser.add_argument(
        "--replace-baseline",
        action="store_true",
        help="Replace this scene's saved normal-room reference with the first frame",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration is not None and args.duration <= 0:
        raise ValueError("Duration must be positive")
    if args.archive_every < 1:
        raise ValueError("Archive interval must be positive")

    roi = RegionOfInterest(*args.roi) if args.roi else None
    capture_config = CaptureConfig(fps=args.fps, roi=roi)
    vision_config = VisionConfig()
    paths = ProjectPaths(root=args.root)
    paths.create_directories()
    memory = MemoryManager(paths.memory)
    detector = AnomalyDetector(vision_config)
    reference_path = memory.reference_path(args.scene)

    if args.replace_baseline or not reference_path.exists():
        reference: Frame | None = None
    else:
        reference = memory.load_reference(args.scene)

    start_time = time.monotonic()
    frame_number = 0
    with ScreenCapture(capture_config) as capture:
        print(
            "Observing the mirrored phone screen without sending keyboard or touch input. "
            "Press Ctrl+C to stop."
        )
        try:
            while args.duration is None or time.monotonic() - start_time < args.duration:
                frame = capture.capture_at_rate()
                frame_number += 1
                if reference is None:
                    reference = frame.copy()
                    memory.save_reference(args.scene, frame)
                    print(f"Saved normal-room baseline: {reference_path}")

                result = detector.compare(frame, reference)
                frame_path: Path | None = None
                if frame_number % args.archive_every == 0:
                    frame_path = paths.screenshots / (
                        f"{args.scene}_{time.time_ns()}_{frame_number:08d}.png"
                    )
                    if not cv2.imwrite(str(frame_path), frame):
                        raise OSError(f"Could not archive screenshot: {frame_path}")
                memory.record_observation(args.scene, result, frame_path)

                if frame_number % max(1, round(args.fps)) == 0:
                    print(
                        f"frames={frame_number} similarity={result.similarity:.3f} "
                        f"changes={len(result.events)} "
                        f"scene_changed={result.scene_changed}"
                    )
        except KeyboardInterrupt:
            print("\nCapture stopped.")


if __name__ == "__main__":
    main()
