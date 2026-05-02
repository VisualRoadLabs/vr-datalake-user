from __future__ import annotations

from google.cloud import storage


class GoogleCloudStorageClient:
    def __init__(self, client: storage.Client | None = None) -> None:
        self.client = client or storage.Client()

    def exists(self, bucket: str, name: str) -> bool:
        return self.client.bucket(bucket).blob(name).exists()

    def list_names(self, bucket: str, prefix: str) -> list[str]:
        return [blob.name for blob in self.client.list_blobs(bucket, prefix=prefix)]

    def download_bytes(self, bucket: str, name: str) -> bytes:
        return self.client.bucket(bucket).blob(name).download_as_bytes()

    def upload_bytes(
        self,
        bucket: str,
        name: str,
        data: bytes,
        content_type: str,
    ) -> None:
        self.client.bucket(bucket).blob(name).upload_from_string(
            data,
            content_type=content_type,
        )

    def copy_blob(
        self,
        source_bucket: str,
        source_name: str,
        destination_bucket: str,
        destination_name: str,
        content_type: str | None = None,
    ) -> None:
        source_bucket_obj = self.client.bucket(source_bucket)
        source = source_bucket_obj.blob(source_name)
        destination_bucket_obj = self.client.bucket(destination_bucket)
        copied = source_bucket_obj.copy_blob(
            source,
            destination_bucket_obj,
            destination_name,
        )
        if content_type:
            copied.content_type = content_type
            copied.patch()
