from datetime import date
from io import BytesIO

from PIL import Image

from anonymizer.batch import BatchAnonymizationJob
from anonymizer.detectors.common import Detection
from anonymizer.service import AnonymizationService
from tests.fakes import FakeDetector, FakeStorage, make_settings


RAW_BUCKET = "bkt-prod-raw-user-usc1"
OUTPUT_BUCKET = "bkt-prod-user-usc1"


def test_batch_processes_only_images_from_process_date_prefix():
    storage = FakeStorage(
        {
            (RAW_BUCKET, "incoming/2026/05/01/veh/image-1.jpg"): _jpg_bytes(),
            (RAW_BUCKET, "incoming/2026/05/01/veh/image-1.json"): b'{"Lines":[]}',
            (RAW_BUCKET, "incoming/2026/05/01/veh/image-2.json"): b'{"Lines":[]}',
            (RAW_BUCKET, "incoming/2026/05/02/veh/image-3.jpg"): _jpg_bytes(),
        }
    )
    settings = make_settings(
        raw_bucket=RAW_BUCKET,
        output_bucket=OUTPUT_BUCKET,
        process_date=date(2026, 5, 1),
    )
    service = AnonymizationService(
        storage=storage,
        detector=FakeDetector([Detection(4, 4, 20, 20, 0.99, "face")]),
        settings=settings,
    )

    result = BatchAnonymizationJob(service=service, settings=settings).run()

    assert result.succeeded
    assert result.discovered_images == 1
    assert result.processed == 1
    assert (OUTPUT_BUCKET, "images/2026/05/image-1.jpg") in storage.objects
    assert (OUTPUT_BUCKET, "labels/2026/05/image-1.json") in storage.objects
    assert (OUTPUT_BUCKET, "images/2026/05/image-3.jpg") not in storage.objects


def _jpg_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(120, 80, 40))
    output = BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()
