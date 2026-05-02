from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


@dataclass(frozen=True)
class Settings:
    raw_bucket: str
    output_bucket: str
    pipeline_timezone: str
    license_plate_detection_model: str | None
    license_plate_model_file: str | None
    license_plate_min_score: float
    face_detector_enabled: bool
    blur_kernel_ratio: float
    min_blur_kernel: int
    bigquery_project: str
    image_metadata_table: str
    privacy_metadata_table: str
    label_review_status_table: str
    label_review_confidence_threshold: float
    process_date: date = field(default_factory=lambda: date(1970, 1, 1))

    @classmethod
    def from_env(cls) -> "Settings":
        config = _load_config(os.getenv("CONFIG_PATH", "configs/prod.yaml"))
        timezone = _required_config(config, "pipeline_timezone")
        return cls(
            raw_bucket=_required_config(config, "raw_bucket"),
            output_bucket=_required_config(config, "output_bucket"),
            process_date=_process_date_from_env(timezone),
            pipeline_timezone=timezone,
            license_plate_detection_model=_nested_config(
                config,
                ["license_plate", "detection_model"],
            )
            or None,
            license_plate_model_file=_nested_config(
                config,
                ["license_plate", "model_file"],
            ),
            license_plate_min_score=float(
                _nested_config(config, ["license_plate", "min_score"])
            ),
            face_detector_enabled=_bool(_required_config(config, "face_detector_enabled")),
            blur_kernel_ratio=float(
                _nested_config(config, ["blur", "kernel_ratio"])
            ),
            min_blur_kernel=int(
                _nested_config(config, ["blur", "min_kernel"])
            ),
            bigquery_project=_nested_config(
                config,
                ["bigquery", "project"],
            ),
            image_metadata_table=_nested_config(
                config,
                ["bigquery", "image_metadata_table"],
            ),
            privacy_metadata_table=_nested_config(
                config,
                ["bigquery", "privacy_metadata_table"],
            ),
            label_review_status_table=_nested_config(
                config,
                ["bigquery", "label_review_status_table"],
            ),
            label_review_confidence_threshold=float(
                _nested_config(
                    config,
                    ["bigquery", "label_review_confidence_threshold"],
                )
            ),
        )


def _process_date_from_env(timezone: str) -> date:
    value = os.getenv("PROCESS_DATE")
    if value:
        return date.fromisoformat(value)

    now = datetime.now(ZoneInfo(timezone))
    return (now - timedelta(days=1)).date()


def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _required_config(config: dict[str, Any], key: str) -> Any:
    if key not in config or config[key] is None:
        raise ValueError(f"Missing required config key: {key}")
    return config[key]


def _nested_config(
    config: dict[str, Any],
    keys: list[str],
) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing required config key: {'.'.join(keys)}")
        current = current[key]
    if current is None:
        raise ValueError(f"Missing required config key: {'.'.join(keys)}")
    return current


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no"}
