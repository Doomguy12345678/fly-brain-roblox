"""Pixel-level anomaly and object-change detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from config import VisionConfig


Frame = NDArray[np.uint8]


class ChangeKind(StrEnum):
    APPEARED = "appeared"
    DISAPPEARED = "disappeared"
    MOVED = "moved"
    SCENE_CHANGE = "scene_change"


@dataclass(frozen=True)
class ChangeEvent:
    kind: ChangeKind
    confidence: float
    current_box: tuple[int, int, int, int] | None = None
    reference_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class DetectionResult:
    difference_map: Frame
    similarity: float
    scene_changed: bool
    events: tuple[ChangeEvent, ...]

    @property
    def has_anomaly(self) -> bool:
        return bool(self.events)


class AnomalyDetector:
    """Compare current frames with a stored normal-room reference."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config

    def compare(self, current: Frame, reference: Frame) -> DetectionResult:
        """Compute a thresholded difference map and changed-object events."""
        if current.shape != reference.shape:
            raise ValueError(
                f"Frame shapes differ: current={current.shape}, reference={reference.shape}"
            )

        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        raw_difference = cv2.absdiff(current_gray, reference_gray)
        _, thresholded = cv2.threshold(
            raw_difference,
            self.config.difference_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        difference_map = cv2.medianBlur(thresholded, 3)
        similarity = self._similarity(current_gray, reference_gray)
        changed_fraction = float(np.count_nonzero(difference_map)) / difference_map.size
        scene_changed = changed_fraction >= self.config.scene_change_threshold

        current_objects = self._objects(current_gray, reference_gray)
        reference_objects = self._objects(reference_gray, current_gray)
        events = self._match_objects(current_objects, reference_objects)
        if scene_changed:
            events.insert(
                0,
                ChangeEvent(
                    kind=ChangeKind.SCENE_CHANGE,
                    confidence=min(1.0, changed_fraction / max(self.config.scene_change_threshold, 1e-6)),
                ),
            )

        return DetectionResult(
            difference_map=difference_map,
            similarity=similarity,
            scene_changed=scene_changed,
            events=tuple(events),
        )

    def match_template(self, image: Frame, template: Frame) -> tuple[float, tuple[int, int]]:
        """Return the best normalized template score and its top-left location."""
        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            raise ValueError("Template dimensions must not exceed image dimensions")
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        scores = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        return float(score), location

    def _objects(self, image: Frame, background: Frame) -> list[tuple[int, int, int, int]]:
        difference = cv2.subtract(image, background)
        gray = (
            cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            if difference.ndim == 3
            else difference
        )
        _, mask = cv2.threshold(
            gray,
            self.config.difference_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [
            cv2.boundingRect(contour)
            for contour in contours
            if cv2.contourArea(contour) >= self.config.object_min_area
        ]
        return boxes

    def _match_objects(
        self,
        current: list[tuple[int, int, int, int]],
        reference: list[tuple[int, int, int, int]],
    ) -> list[ChangeEvent]:
        events: list[ChangeEvent] = []
        unmatched_reference = set(range(len(reference)))
        unmatched_current: list[tuple[int, int, int, int]] = []

        for current_box in current:
            best_index: int | None = None
            best_iou = 0.0
            for index in unmatched_reference:
                overlap = self._intersection_over_union(current_box, reference[index])
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index

            if best_index is None or best_iou < 0.1:
                unmatched_current.append(current_box)
                continue

            reference_box = reference[best_index]
            unmatched_reference.remove(best_index)
            if self._center_distance(current_box, reference_box) >= 8:
                events.append(
                    ChangeEvent(
                        ChangeKind.MOVED,
                        0.7,
                        current_box=current_box,
                        reference_box=reference_box,
                    )
                )

        unmatched_reference_boxes = [reference[index] for index in unmatched_reference]
        while unmatched_current and unmatched_reference_boxes:
            current_box = unmatched_current[0]
            candidates = [
                (
                    self._shape_similarity(current_box, reference_box),
                    index,
                )
                for index, reference_box in enumerate(unmatched_reference_boxes)
            ]
            similarity, reference_index = max(candidates, default=(0.0, -1))
            reference_box = (
                unmatched_reference_boxes[reference_index]
                if reference_index >= 0
                else None
            )
            if (
                reference_box is not None
                and similarity >= 0.6
                and self._center_distance(current_box, reference_box) >= 8
            ):
                events.append(
                    ChangeEvent(
                        ChangeKind.MOVED,
                        similarity,
                        current_box=current_box,
                        reference_box=reference_box,
                    )
                )
                unmatched_current.pop(0)
                unmatched_reference_boxes.pop(reference_index)
            else:
                events.append(
                    ChangeEvent(ChangeKind.APPEARED, 0.7, current_box=current_box)
                )
                unmatched_current.pop(0)

        events.extend(
            ChangeEvent(ChangeKind.APPEARED, 0.7, current_box=box)
            for box in unmatched_current
        )
        events.extend(
            ChangeEvent(ChangeKind.DISAPPEARED, 0.7, reference_box=box)
            for box in unmatched_reference_boxes
        )
        return events

    @staticmethod
    def _shape_similarity(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        first_area = first[2] * first[3]
        second_area = second[2] * second[3]
        area_ratio = min(first_area, second_area) / max(first_area, second_area)
        first_aspect = first[2] / first[3]
        second_aspect = second[2] / second[3]
        aspect_similarity = min(first_aspect, second_aspect) / max(
            first_aspect, second_aspect
        )
        return float((area_ratio + aspect_similarity) / 2)

    @staticmethod
    def _similarity(first: Frame, second: Frame) -> float:
        score = cv2.matchTemplate(first, second, cv2.TM_CCOEFF_NORMED)[0, 0]
        if not np.isfinite(score):
            return 1.0 if np.array_equal(first, second) else 0.0
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _center_distance(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        first_x, first_y = first[0] + first[2] / 2, first[1] + first[3] / 2
        second_x, second_y = second[0] + second[2] / 2, second[1] + second[3] / 2
        return float(np.hypot(first_x - second_x, first_y - second_y))

    @staticmethod
    def _intersection_over_union(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        first_x, first_y, first_w, first_h = first
        second_x, second_y, second_w, second_h = second
        left = max(first_x, second_x)
        top = max(first_y, second_y)
        right = min(first_x + first_w, second_x + second_w)
        bottom = min(first_y + first_h, second_y + second_h)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = first_w * first_h + second_w * second_h - intersection
        return intersection / union if union else 0.0
