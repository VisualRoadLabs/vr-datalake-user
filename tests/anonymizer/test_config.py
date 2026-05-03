from datetime import date

import pytest

from anonymizer.config import Settings


def test_settings_loads_yaml_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
raw_bucket: raw-bucket
output_bucket: output-bucket
pipeline_timezone: UTC
face_detector_enabled: false
license_plate:
  detection_model: morsetechlab/yolov11-license-plate-detection
  model_file: weights.pt
  min_score: 0.42
blur:
  kernel_ratio: 0.2
  min_kernel: 41
bigquery:
  project: test-project
  image_metadata_table: ds.tbl_images
  privacy_metadata_table: ds.tbl_privacy
  label_review_status_table: ds_review.tbl_status
  label_review_confidence_threshold: 0.7
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    monkeypatch.setenv("PROCESS_DATE", "2026-05-01")

    settings = Settings.from_env()

    assert settings.raw_bucket == "raw-bucket"
    assert settings.output_bucket == "output-bucket"
    assert settings.process_date == date(2026, 5, 1)
    assert settings.face_detector_enabled is False
    assert settings.license_plate_detection_model == "morsetechlab/yolov11-license-plate-detection"
    assert settings.license_plate_model_file == "weights.pt"
    assert settings.license_plate_min_score == 0.42
    assert settings.blur_kernel_ratio == 0.2
    assert settings.min_blur_kernel == 41
    assert settings.bigquery_project == "test-project"
    assert settings.image_metadata_table == "ds.tbl_images"
    assert settings.privacy_metadata_table == "ds.tbl_privacy"
    assert settings.label_review_status_table == "ds_review.tbl_status"
    assert settings.label_review_confidence_threshold == 0.7


def test_yaml_config_is_not_overridden_by_legacy_env_vars(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
raw_bucket: raw-bucket
output_bucket: output-bucket
pipeline_timezone: UTC
face_detector_enabled: true
license_plate:
  detection_model: model
  model_file: weights.pt
  min_score: 0.25
blur:
  kernel_ratio: 0.18
  min_kernel: 31
bigquery:
  project: test-project
  image_metadata_table: ds.tbl_images
  privacy_metadata_table: ds.tbl_privacy
  label_review_status_table: ds_review.tbl_status
  label_review_confidence_threshold: 0.8
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    monkeypatch.setenv("RAW_BUCKET", "override-raw")
    monkeypatch.setenv("PROCESS_DATE", "2026-05-01")

    settings = Settings.from_env()

    assert settings.raw_bucket == "raw-bucket"
    assert settings.process_date == date(2026, 5, 1)


def test_missing_required_yaml_key_fails(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("raw_bucket: raw-bucket\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    with pytest.raises(ValueError, match="Missing required config key: pipeline_timezone"):
        Settings.from_env()
