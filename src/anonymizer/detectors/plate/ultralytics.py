from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anonymizer.detectors.common import Detection
from anonymizer.detectors.env import configure_detector_environment

configure_detector_environment()


class UltralyticsHuggingFaceLicensePlateDetector:
    def __init__(
        self,
        repo_id: str,
        model_file: str | None,
        min_score: float,
    ) -> None:
        self.repo_id = repo_id
        self.model_file = model_file
        self.min_score = min_score
        self._model = None

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        image_path = _write_temp_jpeg(image_rgb)
        try:
            results = self._load_model().predict(
                source=image_path,
                conf=self.min_score,
                verbose=False,
            )
            return parse_ultralytics_predictions(results, self.min_score)
        finally:
            Path(image_path).unlink(missing_ok=True)

    def _load_model(self):
        if self._model is None:
            from huggingface_hub import hf_hub_download
            from ultralytics import YOLO

            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.model_file,
            )
            self._model = YOLO(model_path)
        return self._model


def parse_ultralytics_predictions(
    predictions: Any,
    min_score: float,
) -> list[Detection]:
    detections: list[Detection] = []

    for result in _prediction_rows(predictions):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        for box, score in zip(xyxy, confidences):
            score = float(score)
            if score < min_score:
                continue
            values = box.tolist() if hasattr(box, "tolist") else box
            x1, y1, x2, y2 = [int(round(value)) for value in values]
            detections.append(Detection(x1, y1, x2, y2, score, "license_plate"))

    return detections


def _write_temp_jpeg(image_rgb: np.ndarray) -> str:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name

        Image.fromarray(image_rgb).save(temp_path, format="JPEG")
        return temp_path
    except Exception:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        raise


def _prediction_rows(predictions: Any) -> list[Any]:
    if predictions is None:
        return []
    if isinstance(predictions, dict):
        for key in ("detections", "predictions", "boxes", "results"):
            if key in predictions:
                return _prediction_rows(predictions[key])
        return [predictions]
    if isinstance(predictions, (list, tuple)):
        return list(predictions)
    return [predictions]
