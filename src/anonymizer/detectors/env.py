from __future__ import annotations

import os


def configure_detector_environment() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
