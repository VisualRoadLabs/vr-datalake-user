from __future__ import annotations

from anonymizer.detectors.common import CompositeDetector
from anonymizer.detectors.face.retinaface import RetinaFaceDetector
from anonymizer.detectors.plate.ultralytics import (
    UltralyticsHuggingFaceLicensePlateDetector,
)


def build_default_detector(
    license_plate_detection_model: str | None,
    license_plate_model_file: str | None,
    license_plate_min_score: float,
    face_detector_enabled: bool,
) -> CompositeDetector:
    detectors = []
    if face_detector_enabled:
        detectors.append(RetinaFaceDetector())
    if license_plate_detection_model:
        detectors.append(
            UltralyticsHuggingFaceLicensePlateDetector(
                license_plate_detection_model,
                model_file=license_plate_model_file,
                min_score=license_plate_min_score,
            )
        )
    return CompositeDetector(detectors)
