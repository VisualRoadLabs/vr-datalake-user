from google.api_core.exceptions import PreconditionFailed

from gcs.storage import GoogleCloudStorageClient, ObjectAlreadyExists


def test_copy_blob_uses_source_bucket_copy_blob():
    fake_client = FakeStorageClient()
    storage = GoogleCloudStorageClient(client=fake_client)

    storage.copy_blob(
        "raw-bucket",
        "incoming/image.json",
        "output-bucket",
        "labels/image.json",
        content_type="application/json",
    )

    source_bucket = fake_client.buckets["raw-bucket"]
    copied_blob = source_bucket.copied[0]
    assert copied_blob.source.name == "incoming/image.json"
    assert copied_blob.destination_bucket.name == "output-bucket"
    assert copied_blob.destination_name == "labels/image.json"
    assert copied_blob.patched is False
    assert copied_blob.if_generation_match == 0


def test_upload_raises_object_already_exists_on_precondition_failure():
    fake_client = FakeStorageClient()
    fake_client.buckets["output-bucket"] = FakeBucket(
        "output-bucket",
        fail_upload=True,
    )
    storage = GoogleCloudStorageClient(client=fake_client)

    try:
        storage.upload_bytes(
            "output-bucket",
            "images/image.jpg",
            b"image",
            content_type="image/jpeg",
        )
    except ObjectAlreadyExists as exc:
        assert exc.bucket == "output-bucket"
        assert exc.name == "images/image.jpg"
    else:
        raise AssertionError("Expected ObjectAlreadyExists")


class FakeStorageClient:
    def __init__(self) -> None:
        self.buckets = {}

    def bucket(self, name: str):
        if name not in self.buckets:
            self.buckets[name] = FakeBucket(name)
        return self.buckets[name]


class FakeBucket:
    def __init__(self, name: str, fail_upload: bool = False) -> None:
        self.name = name
        self.fail_upload = fail_upload
        self.copied = []

    def blob(self, name: str):
        return FakeBlob(name, fail_upload=self.fail_upload)

    def copy_blob(
        self,
        source,
        destination_bucket,
        destination_name,
        if_generation_match=None,
    ):
        copied = FakeCopiedBlob(
            source,
            destination_bucket,
            destination_name,
            if_generation_match,
        )
        self.copied.append(copied)
        return copied


class FakeBlob:
    def __init__(self, name: str, fail_upload: bool = False) -> None:
        self.name = name
        self.fail_upload = fail_upload

    def upload_from_string(
        self,
        data: bytes,
        content_type: str,
        if_generation_match=None,
    ) -> None:
        if self.fail_upload:
            raise PreconditionFailed("already exists")


class FakeCopiedBlob:
    def __init__(
        self,
        source,
        destination_bucket,
        destination_name,
        if_generation_match,
    ) -> None:
        self.source = source
        self.destination_bucket = destination_bucket
        self.destination_name = destination_name
        self.if_generation_match = if_generation_match
        self.content_type = None
        self.patched = False

    def patch(self) -> None:
        self.patched = True
