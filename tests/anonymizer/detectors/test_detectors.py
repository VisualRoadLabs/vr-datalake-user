from anonymizer.detectors.common import Detection
from anonymizer.detectors.plate.ultralytics import parse_ultralytics_predictions


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10.2, 20.6, 40.1, 50.9], [1, 2, 3, 4]])
        self.conf = FakeTensor([0.91, 0.12])


class FakeResult:
    boxes = FakeBoxes()


def test_parse_ultralytics_predictions_filters_by_score():
    detections = parse_ultralytics_predictions([FakeResult()], min_score=0.35)

    assert len(detections) == 1
    assert detections[0] == Detection(10, 21, 40, 51, 0.91, "license_plate")
