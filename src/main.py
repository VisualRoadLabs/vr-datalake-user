from __future__ import annotations

import json
import logging
import sys

from anonymizer.batch import BatchAnonymizationJob
from anonymizer.config import Settings
from anonymizer.detectors.factory import build_default_detector
from anonymizer.metadata import GoogleBigQueryMetadataWriter
from anonymizer.service import AnonymizationService
from gcs.storage import GoogleCloudStorageClient


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    detector = build_default_detector(
        license_plate_detection_model=settings.license_plate_detection_model,
        license_plate_model_file=settings.license_plate_model_file,
        license_plate_min_score=settings.license_plate_min_score,
        face_detector_enabled=settings.face_detector_enabled,
    )
    service = AnonymizationService(
        storage=GoogleCloudStorageClient(),
        detector=detector,
        settings=settings,
        metadata_writer=GoogleBigQueryMetadataWriter(settings),
    )
    result = BatchAnonymizationJob(service=service, settings=settings).run()
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
