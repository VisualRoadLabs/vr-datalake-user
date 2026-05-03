from io import BytesIO

from PIL import Image

from anonymizer.detectors.common import Detection
from anonymizer.service import AnonymizationService
from gcs.storage import ObjectAlreadyExists
from tests.fakes import FakeDetector, FakeMetadataWriter, FakeStorage, make_settings


RAW_BUCKET = "bkt-prod-raw-user-usc1"
OUTPUT_BUCKET = "bkt-prod-user-usc1"


def test_image_object_uploads_anonymized_image_and_copies_json():
    storage = FakeStorage(
        {
            (
                RAW_BUCKET,
                "incoming/2026/05/02/vehicle/image-1.jpg",
            ): _jpg_bytes(),
            (
                RAW_BUCKET,
                "incoming/2026/05/02/vehicle/image-1.json",
            ): b'{"confidence_score":0.91,"Lines":[]}',
        }
    )
    service = _service(storage, [Detection(4, 4, 20, 20, 0.99, "face")])

    result = service.process_raw_image(
        RAW_BUCKET,
        "incoming/2026/05/02/vehicle/image-1.jpg",
    )

    assert result.status == "processed"
    assert (OUTPUT_BUCKET, "images/2026/05/image-1.jpg") in storage.objects
    assert (OUTPUT_BUCKET, "labels/2026/05/image-1.json") in storage.objects
    assert storage.copies[0][3] == "labels/2026/05/image-1.json"


def test_high_confidence_json_writes_image_and_privacy_metadata_without_review():
    metadata_writer = FakeMetadataWriter()
    storage = FakeStorage(
        {
            (
                RAW_BUCKET,
                "incoming/2026/05/02/vehiclehash/image-1.jpg",
            ): _jpg_bytes(),
            (
                RAW_BUCKET,
                "incoming/2026/05/02/vehiclehash/image-1.json",
            ): (
                b'{"confidence_score":0.91,'
                b'"model_version_at_capture":"lane-v1","Lines":[]}'
            ),
        }
    )
    service = _service(storage, [], metadata_writer=metadata_writer)

    result = service.process_raw_image(
        RAW_BUCKET,
        "incoming/2026/05/02/vehiclehash/image-1.jpg",
    )

    assert result.status == "processed"
    rows = metadata_writer.rows[0]
    assert rows.image["image_id"] == "image-1"
    assert rows.image["source_type"] == "user"
    assert rows.image["gcs_uri"] == "gs://bkt-prod-user-usc1/images/2026/05/image-1.jpg"
    assert rows.image["label_gcs_uri"] == "gs://bkt-prod-user-usc1/labels/2026/05/image-1.json"
    assert rows.privacy["vehicle_id_hash"] == "vehiclehash"
    assert rows.privacy["confidence_score"] == 0.91
    assert rows.privacy["dlp_status"] == "processed"
    assert rows.review_status is None


def test_low_confidence_json_writes_review_pending_metadata():
    metadata_writer = FakeMetadataWriter()
    storage = FakeStorage(
        {
            (
                RAW_BUCKET,
                "incoming/2026/05/02/vehiclehash/image-2.jpg",
            ): _jpg_bytes(),
            (
                RAW_BUCKET,
                "incoming/2026/05/02/vehiclehash/image-2.json",
            ): (
                b'{"confidence_score":0.41,'
                b'"model_version_at_capture":"lane-v2","Lines":[]}'
            ),
        }
    )
    service = _service(storage, [], metadata_writer=metadata_writer)

    result = service.process_raw_image(
        RAW_BUCKET,
        "incoming/2026/05/02/vehiclehash/image-2.jpg",
    )

    assert result.status == "processed"
    rows = metadata_writer.rows[0]
    assert rows.privacy["confidence_score"] == 0.41
    assert rows.review_status == {
        "image_id": "image-2",
        "status": "review_pending",
        "model_version": "lane-v2",
        "platform_task_id": None,
        "reviewed_by": None,
        "reviewed_at": None,
        "skip_reason": None,
        "created_at": rows.review_status["created_at"],
    }


def test_missing_json_writes_null_label_metadata_without_review():
    metadata_writer = FakeMetadataWriter()
    storage = FakeStorage(
        {
            (
                RAW_BUCKET,
                "incoming/2026/05/02/vehiclehash/image-3.jpg",
            ): _jpg_bytes(),
        }
    )
    service = _service(storage, [], metadata_writer=metadata_writer)

    result = service.process_raw_image(
        RAW_BUCKET,
        "incoming/2026/05/02/vehiclehash/image-3.jpg",
    )

    assert result.status == "processed"
    rows = metadata_writer.rows[0]
    assert rows.image["label_gcs_uri"] is None
    assert rows.privacy["confidence_score"] is None
    assert rows.privacy["dlp_status"] == "processed"
    assert rows.review_status is None


def test_image_object_without_json_uploads_only_image():
    storage = FakeStorage(
        {
            (
                RAW_BUCKET,
                "incoming/2026/05/02/vehicle/image-2.jpg",
            ): _jpg_bytes(),
        }
    )
    service = _service(storage, [])

    result = service.process_raw_image(
        RAW_BUCKET,
        "incoming/2026/05/02/vehicle/image-2.jpg",
    )

    assert result.status == "processed"
    assert (OUTPUT_BUCKET, "images/2026/05/image-2.jpg") in storage.objects
    assert (OUTPUT_BUCKET, "labels/2026/05/image-2.json") not in storage.objects


def test_existing_output_image_logs_warning_and_does_not_fail(caplog):
    storage = ExistingOutputStorage(
        {
            (
                RAW_BUCKET,
                "incoming/2026/05/02/vehicle/image-4.jpg",
            ): _jpg_bytes(),
        }
    )
    service = _service(storage, [])

    result = service.process_raw_image(
        RAW_BUCKET,
        "incoming/2026/05/02/vehicle/image-4.jpg",
    )

    assert result.status == "processed"
    assert "Output image already exists" in caplog.text


def _service(
    storage: FakeStorage,
    detections: list[Detection],
    metadata_writer=None,
) -> AnonymizationService:
    return AnonymizationService(
        storage=storage,
        detector=FakeDetector(detections),
        settings=make_settings(raw_bucket=RAW_BUCKET, output_bucket=OUTPUT_BUCKET),
        metadata_writer=metadata_writer,
    )


def _jpg_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(120, 80, 40))
    output = BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


class ExistingOutputStorage(FakeStorage):
    def upload_bytes(
        self,
        bucket: str,
        name: str,
        data: bytes,
        content_type: str,
    ) -> None:
        raise ObjectAlreadyExists(bucket, name)
