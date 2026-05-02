from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    label: str

    def clamped(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            max(0, min(self.x1, width)),
            max(0, min(self.y1, height)),
            max(0, min(self.x2, width)),
            max(0, min(self.y2, height)),
        )


class CompositeDetector:
    def __init__(self, detectors: list[Any]) -> None:
        self.detectors = detectors

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for detector in self.detectors:
            detections.extend(detector.detect(image_rgb))
        return detections
