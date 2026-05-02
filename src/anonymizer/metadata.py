from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from anonymizer.config import Settings
from anonymizer.paths import RawObjectPath


REVIEW_PENDING = "review_pending"


@dataclass(frozen=True)
class ImageMetadata:
    width_px: int
    height_px: int
    image_format: str
    size_bytes: int
    md5_hash: str


@dataclass(frozen=True)
class VehicleLabelMetadata:
    confidence_score: float | None = None
    model_version_at_capture: str | None = None
    received_at: str | None = None
    consent_version: str | None = None


@dataclass(frozen=True)
class BigQueryRows:
    image: dict[str, Any]
    privacy: dict[str, Any]
    review_status: dict[str, Any] | None = None


class GoogleBigQueryMetadataWriter:
    def __init__(self, settings: Settings) -> None:
        from google.cloud import bigquery

        self.client = bigquery.Client(project=settings.bigquery_project)
        self.settings = settings

    def write(self, rows: BigQueryRows) -> None:
        self._insert(self.settings.image_metadata_table, rows.image)
        self._insert(self.settings.privacy_metadata_table, rows.privacy)
        if rows.review_status is not None:
            self._insert(self.settings.label_review_status_table, rows.review_status)

    def _insert(self, table: str, row: dict[str, Any]) -> None:
        table_id = _table_id(self.settings.bigquery_project, table)
        errors = self.client.insert_rows_json(
            table_id,
            [row],
            ignore_unknown_values=True,
        )
        if errors:
            raise RuntimeError(f"BigQuery insert failed for {table_id}: {errors}")


def build_bigquery_rows(
    raw_path: RawObjectPath,
    settings: Settings,
    image_metadata: ImageMetadata,
    label_metadata: VehicleLabelMetadata | None,
    output_label_name: str | None,
    now: datetime | None = None,
) -> BigQueryRows:
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    label_gcs_uri = (
        _gcs_uri(settings.output_bucket, output_label_name)
        if output_label_name is not None
        else None
    )
    confidence_score = (
        label_metadata.confidence_score if label_metadata is not None else None
    )
    model_version = (
        label_metadata.model_version_at_capture if label_metadata is not None else None
    )

    image_row = {
        "image_id": raw_path.image_id,
        "source_type": "user",
        "source_dataset_id": None,
        "gcs_uri": _gcs_uri(settings.output_bucket, raw_path.output_image_name),
        "label_gcs_uri": label_gcs_uri,
        "original_filename": None,
        "original_relative_path": None,
        "width_px": image_metadata.width_px,
        "height_px": image_metadata.height_px,
        "format": image_metadata.image_format,
        "size_bytes": image_metadata.size_bytes,
        "md5_hash": image_metadata.md5_hash,
        "created_at": timestamp,
        "gcs_status": "available",
        "default_split": None,
    }

    privacy_row = {
        "image_id": raw_path.image_id,
        "vehicle_id_hash": raw_path.vehicle_id_hash,
        "model_version_at_capture": model_version,
        "confidence_score": confidence_score,
        "received_at": _received_at(label_metadata, timestamp),
        "dlp_status": "processed",
        "dlp_processed_at": timestamp,
        "gcs_raw_uri": _gcs_uri(settings.raw_bucket, raw_path.sibling_image_name),
    }
    if label_metadata is not None and label_metadata.consent_version is not None:
        privacy_row["consent_version"] = label_metadata.consent_version

    review_row = None
    if _requires_review(confidence_score, output_label_name, settings):
        review_row = {
            "image_id": raw_path.image_id,
            "status": REVIEW_PENDING,
            "model_version": model_version,
            "platform_task_id": None,
            "reviewed_by": None,
            "reviewed_at": None,
            "skip_reason": None,
            "created_at": timestamp,
        }

    return BigQueryRows(
        image=image_row,
        privacy=privacy_row,
        review_status=review_row,
    )


def image_metadata_from_bytes(image_rgb: Any, image_bytes: bytes) -> ImageMetadata:
    height_px, width_px = image_rgb.shape[:2]
    return ImageMetadata(
        width_px=int(width_px),
        height_px=int(height_px),
        image_format="jpg",
        size_bytes=len(image_bytes),
        md5_hash=hashlib.md5(image_bytes).hexdigest(),
    )


def label_metadata_from_json(data: dict[str, Any]) -> VehicleLabelMetadata:
    return VehicleLabelMetadata(
        confidence_score=_optional_float(data.get("confidence_score")),
        model_version_at_capture=_optional_string(data.get("model_version_at_capture")),
        received_at=_optional_string(data.get("received_at")),
        consent_version=_optional_string(data.get("consent_version")),
    )


def _requires_review(
    confidence_score: float | None,
    output_label_name: str | None,
    settings: Settings,
) -> bool:
    if output_label_name is None or confidence_score is None:
        return False
    return confidence_score < settings.label_review_confidence_threshold


def _received_at(
    label_metadata: VehicleLabelMetadata | None,
    default: str,
) -> str:
    if label_metadata is None or not label_metadata.received_at:
        return default
    return label_metadata.received_at


def _gcs_uri(bucket: str, name: str | None) -> str | None:
    if name is None:
        return None
    return f"gs://{bucket}/{name}"


def _table_id(project: str, table: str) -> str:
    if table.count(".") == 2:
        return table
    return f"{project}.{table}"


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
