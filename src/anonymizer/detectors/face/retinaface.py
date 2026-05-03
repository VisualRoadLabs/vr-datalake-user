from __future__ import annotations

import numpy as np

from anonymizer.detectors.common import Detection
from anonymizer.detectors.env import configure_detector_environment

configure_detector_environment()


class RetinaFaceDetector:
    def __init__(self, min_score: float = 0.8) -> None:
        self.min_score = min_score

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        try:
            from retinaface import RetinaFace
        except ImportError as exc:
            raise RuntimeError(
                "RetinaFace is enabled but the 'retinaface' package is not installed."
            ) from exc

        faces = RetinaFace.detect_faces(image_rgb)
        if not isinstance(faces, dict):
            return []

        detections: list[Detection] = []
        for face in faces.values():
            score = float(face.get("score", 0.0))
            if score < self.min_score:
                continue
            x1, y1, x2, y2 = [int(v) for v in face["facial_area"]]
            detections.append(Detection(x1, y1, x2, y2, score, "face"))
        return detections
