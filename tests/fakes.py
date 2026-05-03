from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from anonymizer.config import Settings
from anonymizer.detectors.common import Detection


def make_settings(
    raw_bucket: str = "bkt-prod-raw-user-usc1",
    output_bucket: str = "bkt-prod-user-usc1",
    process_date: date = date(2026, 5, 1),
) -> Settings:
    return Settings(
        raw_bucket=raw_bucket,
        output_bucket=output_bucket,
        pipeline_timezone="Europe/Madrid",
        license_plate_detection_model="morsetechlab/yolov11-license-plate-detection",
        license_plate_model_file="license-plate-finetune-v1n.pt",
        license_plate_min_score=0.25,
        face_detector_enabled=True,
        blur_kernel_ratio=0.18,
        min_blur_kernel=31,
        bigquery_project="vr-prj-prod-data-v1",
        image_metadata_table="ds_raw_metadata.tbl_images",
        privacy_metadata_table="ds_raw_metadata.tbl_user_images_privacy",
        label_review_status_table="ds_label_review.tbl_label_review_status",
        label_review_confidence_threshold=0.8,
        process_date=process_date,
    )


class FakeStorage:
    def __init__(self, objects: dict[tuple[str, str], bytes] | None = None) -> None:
        self.objects = objects or {}
        self.uploads: list[tuple[str, str, bytes, str]] = []
        self.copies: list[tuple[str, str, str, str, str | None]] = []

    def exists(self, bucket: str, name: str) -> bool:
        return (bucket, name) in self.objects

    def list_names(self, bucket: str, prefix: str) -> list[str]:
        return sorted(
            name
            for object_bucket, name in self.objects
            if object_bucket == bucket and name.startswith(prefix)
        )

    def download_bytes(self, bucket: str, name: str) -> bytes:
        return self.objects[(bucket, name)]

    def upload_bytes(
        self,
        bucket: str,
        name: str,
        data: bytes,
        content_type: str,
    ) -> None:
        self.objects[(bucket, name)] = data
        self.uploads.append((bucket, name, data, content_type))

    def copy_blob(
        self,
        source_bucket: str,
        source_name: str,
        destination_bucket: str,
        destination_name: str,
        content_type: str | None = None,
    ) -> None:
        self.objects[(destination_bucket, destination_name)] = self.objects[
            (source_bucket, source_name)
        ]
        self.copies.append(
            (
                source_bucket,
                source_name,
                destination_bucket,
                destination_name,
                content_type,
            )
        )


@dataclass
class FakeDetector:
    detections: list[Detection]

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        return self.detections


class FakeMetadataWriter:
    def __init__(self) -> None:
        self.rows = []

    def write(self, rows) -> None:
        self.rows.append(rows)
