from __future__ import annotations

import logging
from dataclasses import dataclass, field

from anonymizer.config import Settings
from anonymizer.paths import IMAGE_SUFFIXES
from anonymizer.service import AnonymizationService, ProcessResult

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchResult:
    process_date: str
    raw_prefix: str
    discovered_images: int
    processed: int = 0
    ignored: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.failed == 0


class BatchAnonymizationJob:
    def __init__(self, service: AnonymizationService, settings: Settings) -> None:
        self.service = service
        self.settings = settings

    def run(self) -> BatchResult:
        prefix = self._raw_prefix()
        image_names = self._list_raw_images(prefix)

        processed = 0
        ignored = 0
        failures: list[str] = []

        LOGGER.info(
            "Starting user image anonymization batch for gs://%s/%s",
            self.settings.raw_bucket,
            prefix,
        )

        for name in image_names:
            try:
                result = self.service.process_raw_image(self.settings.raw_bucket, name)
                if result.status == "processed":
                    processed += 1
                else:
                    ignored += 1
                LOGGER.info("Processed %s: %s", name, result)
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                LOGGER.exception("Failed processing %s", name)

        return BatchResult(
            process_date=self.settings.process_date.isoformat(),
            raw_prefix=prefix,
            discovered_images=len(image_names),
            processed=processed,
            ignored=ignored,
            failed=len(failures),
            failures=failures,
        )

    def _raw_prefix(self) -> str:
        return (
            "incoming/"
            f"{self.settings.process_date:%Y}/"
            f"{self.settings.process_date:%m}/"
            f"{self.settings.process_date:%d}/"
        )

    def _list_raw_images(self, prefix: str) -> list[str]:
        names = self.service.storage.list_names(self.settings.raw_bucket, prefix)
        return sorted(
            name
            for name in names
            if any(name.lower().endswith(suffix) for suffix in IMAGE_SUFFIXES)
        )
