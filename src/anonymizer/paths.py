from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


IMAGE_SUFFIXES = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class RawObjectPath:
    year: str
    month: str
    day: str
    vehicle_id_hash: str
    session_id: str
    image_id: str
    source_image_name: str
    raw_parent_prefix: str

    @property
    def sibling_image_name(self) -> str:
        return self.source_image_name

    @property
    def sibling_json_name(self) -> str:
        return f"{self.raw_parent_prefix}/{self.image_id}.json"

    @property
    def output_image_name(self) -> str:
        return f"images/{self.vehicle_id_hash}/{self.session_id}/{self.image_id}.jpg"

    @property
    def output_label_name(self) -> str:
        return f"labels/{self.vehicle_id_hash}/{self.session_id}/{self.image_id}.json"


def parse_raw_image_name(name: str) -> RawObjectPath | None:
    path = PurePosixPath(name)
    parts = path.parts
    if len(parts) != 7 or parts[0] != "incoming":
        return None

    stem = path.stem
    suffix = path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        return None

    year, month, day = parts[1], parts[2], parts[3]
    vehicle_id_hash = parts[4]
    session_id = parts[5]
    raw_parent_prefix = f"incoming/{year}/{month}/{day}/{vehicle_id_hash}/{session_id}"

    return RawObjectPath(
        year=year,
        month=month,
        day=day,
        vehicle_id_hash=vehicle_id_hash,
        session_id=session_id,
        image_id=stem,
        source_image_name=name,
        raw_parent_prefix=raw_parent_prefix,
    )
