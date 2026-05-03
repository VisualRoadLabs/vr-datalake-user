from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from anonymizer.blur import anonymize_image_bytes, decode_image
from anonymizer.config import Settings
from anonymizer.metadata import (
    build_bigquery_rows,
    image_metadata_from_bytes,
    label_metadata_from_json,
)
from anonymizer.paths import RawObjectPath, parse_raw_image_name
from gcs.storage import ObjectAlreadyExists


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessResult:
    status: str
    detail: str
    output_image: str | None = None
    output_label: str | None = None


class AnonymizationService:
    def __init__(
        self,
        storage: Any,
        detector: Any,
        settings: Settings,
        metadata_writer: Any | None = None,
    ) -> None:
        self.storage = storage
        self.detector = detector
        self.settings = settings
        self.metadata_writer = metadata_writer

    def process_raw_image(self, bucket: str, name: str) -> ProcessResult:
        if bucket != self.settings.raw_bucket:
            return ProcessResult("ignored", f"Bucket {bucket} is not configured raw bucket")

        raw_path = parse_raw_image_name(name)
        if raw_path is None:
            return ProcessResult("ignored", f"Object {name} is not a raw image")

        return self._process_image(raw_path)

    def _process_image(self, raw_path: RawObjectPath) -> ProcessResult:
        raw_image = self.storage.download_bytes(
            self.settings.raw_bucket,
            raw_path.sibling_image_name,
        )
        image_rgb = decode_image(raw_image)
        detections = self.detector.detect(image_rgb)
        anonymized = anonymize_image_bytes(
            raw_image,
            detections,
            blur_kernel_ratio=self.settings.blur_kernel_ratio,
            min_blur_kernel=self.settings.min_blur_kernel,
        )

        self._upload_image(raw_path, anonymized)

        output_label, label_metadata = self._copy_label_if_present(raw_path)
        self._write_metadata(
            raw_path=raw_path,
            image_rgb=image_rgb,
            anonymized_image=anonymized,
            output_label=output_label,
            label_metadata=label_metadata,
        )
        return ProcessResult(
            "processed",
            f"Anonymized image with {len(detections)} PII detections",
            output_image=raw_path.output_image_name,
            output_label=output_label,
        )

    def _upload_image(self, raw_path: RawObjectPath, anonymized: bytes) -> None:
        try:
            self.storage.upload_bytes(
                self.settings.output_bucket,
                raw_path.output_image_name,
                anonymized,
                content_type="image/jpeg",
            )
        except ObjectAlreadyExists:
            LOGGER.warning(
                "Output image already exists, keeping existing object: gs://%s/%s",
                self.settings.output_bucket,
                raw_path.output_image_name,
            )

    def _copy_label_if_present(self, raw_path: RawObjectPath) -> tuple[str | None, Any | None]:
        if not self.storage.exists(self.settings.raw_bucket, raw_path.sibling_json_name):
            return None, None

        label_bytes = self.storage.download_bytes(
            self.settings.raw_bucket,
            raw_path.sibling_json_name,
        )
        label_metadata = label_metadata_from_json(json.loads(label_bytes.decode("utf-8")))

        try:
            self.storage.copy_blob(
                self.settings.raw_bucket,
                raw_path.sibling_json_name,
                self.settings.output_bucket,
                raw_path.output_label_name,
                content_type="application/json",
            )
        except ObjectAlreadyExists:
            LOGGER.warning(
                "Output label already exists, keeping existing object: gs://%s/%s",
                self.settings.output_bucket,
                raw_path.output_label_name,
            )
        return raw_path.output_label_name, label_metadata

    def _write_metadata(
        self,
        raw_path: RawObjectPath,
        image_rgb: Any,
        anonymized_image: bytes,
        output_label: str | None,
        label_metadata: Any | None,
    ) -> None:
        if self.metadata_writer is None:
            return

        rows = build_bigquery_rows(
            raw_path=raw_path,
            settings=self.settings,
            image_metadata=image_metadata_from_bytes(image_rgb, anonymized_image),
            label_metadata=label_metadata,
            output_label_name=output_label,
        )
        self.metadata_writer.write(rows)
