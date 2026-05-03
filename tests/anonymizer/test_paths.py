from anonymizer.paths import parse_raw_image_name


def test_parse_raw_image_path_maps_output_locations():
    parsed = parse_raw_image_name(
        "incoming/2026/05/02/vehiclehash/image-123.jpg"
    )

    assert parsed is not None
    assert parsed.sibling_image_name == "incoming/2026/05/02/vehiclehash/image-123.jpg"
    assert parsed.sibling_json_name == "incoming/2026/05/02/vehiclehash/image-123.json"
    assert parsed.output_image_name == "images/2026/05/image-123.jpg"
    assert parsed.output_label_name == "labels/2026/05/image-123.json"


def test_parse_preserves_source_extension_for_download():
    parsed = parse_raw_image_name("incoming/2026/05/02/vehiclehash/image-123.jpeg")

    assert parsed is not None
    assert parsed.sibling_image_name == "incoming/2026/05/02/vehiclehash/image-123.jpeg"
    assert parsed.output_image_name == "images/2026/05/image-123.jpg"


def test_parse_ignores_image_without_vehicle_folder():
    assert parse_raw_image_name("incoming/2026/05/02/image-123.jpg") is None


def test_parse_ignores_non_incoming_layout():
    assert parse_raw_image_name("other/2026/05/image.jpg") is None
