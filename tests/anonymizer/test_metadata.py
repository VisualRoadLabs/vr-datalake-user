from anonymizer.metadata import BigQueryRows, GoogleBigQueryMetadataWriter
from tests.fakes import make_settings


def test_bigquery_writer_skips_inserts_when_image_exists(caplog):
    writer = _writer(existing=True)

    writer.write(_rows())

    assert writer.client.inserted == []
    assert "BigQuery metadata already exists for image_id=image-1" in caplog.text


def test_bigquery_writer_inserts_rows_when_image_is_new():
    writer = _writer(existing=False)

    writer.write(_rows(review_status={"image_id": "image-1"}))

    assert [table for table, _ in writer.client.inserted] == [
        "vr-prj-prod-data-v1.ds_raw_metadata.tbl_images",
        "vr-prj-prod-data-v1.ds_raw_metadata.tbl_user_images_privacy",
        "vr-prj-prod-data-v1.ds_label_review.tbl_label_review_status",
    ]


def _writer(existing: bool) -> GoogleBigQueryMetadataWriter:
    writer = object.__new__(GoogleBigQueryMetadataWriter)
    writer.bigquery = FakeBigQueryModule()
    writer.client = FakeBigQueryClient(existing=existing)
    writer.settings = make_settings()
    return writer


def _rows(review_status=None) -> BigQueryRows:
    return BigQueryRows(
        image={"image_id": "image-1"},
        privacy={"image_id": "image-1"},
        review_status=review_status,
    )


class FakeBigQueryModule:
    class QueryJobConfig:
        def __init__(self, query_parameters):
            self.query_parameters = query_parameters

    class ScalarQueryParameter:
        def __init__(self, name, parameter_type, value):
            self.name = name
            self.parameter_type = parameter_type
            self.value = value


class FakeBigQueryClient:
    def __init__(self, existing: bool) -> None:
        self.existing = existing
        self.inserted = []

    def query(self, query, job_config):
        return FakeQueryJob(existing=self.existing)

    def insert_rows_json(self, table_id, rows, ignore_unknown_values):
        self.inserted.append((table_id, rows[0]))
        return []


class FakeQueryJob:
    def __init__(self, existing: bool) -> None:
        self.existing = existing

    def result(self):
        return [object()] if self.existing else []
