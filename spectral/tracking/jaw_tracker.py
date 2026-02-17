from typing import List, Optional

from spectral.models import Point2D, JawData, TrackingConfig


class JawTracker:
    JAW_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]

    LOWER_JAW = [
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132,
    ]

    CHIN = 152
    JAW_LEFT = 234
    JAW_RIGHT = 454
    FOREHEAD_CENTER = 10
    NOSE_TIP = 1

    UPPER_LIP = 13
    LOWER_LIP = 14

    def __init__(self, config: TrackingConfig):
        self._config = config
        self._baseline_openness = None

    def track(self, landmarks: List[Point2D]) -> Optional[JawData]:
        if len(landmarks) < 468:
            return None

        jaw_landmarks = [landmarks[i] for i in self.JAW_INDICES]
        lower_jaw = [landmarks[i] for i in self.LOWER_JAW]

        cx = sum(p.x for p in lower_jaw) / len(lower_jaw)
        cy = sum(p.y for p in lower_jaw) / len(lower_jaw)
        center = Point2D(cx, cy)

        width = landmarks[self.JAW_LEFT].distance_to(landmarks[self.JAW_RIGHT])
        openness = self._compute_openness(landmarks)
        deviation = self._compute_deviation(landmarks)
        clench_amount = self._compute_clench(landmarks)

        return JawData(
            landmarks=jaw_landmarks,
            center=center,
            openness=openness,
            width=width,
            deviation=deviation,
            clench_amount=clench_amount,
        )

    def _compute_openness(self, landmarks: List[Point2D]) -> float:
        chin = landmarks[self.CHIN]
        forehead = landmarks[self.FOREHEAD_CENTER]
        face_height = forehead.distance_to(chin)

        upper_lip = landmarks[self.UPPER_LIP]
        lower_lip = landmarks[self.LOWER_LIP]
        lip_distance = upper_lip.distance_to(lower_lip)

        if face_height < 1e-6:
            return 0.0

        ratio = lip_distance / face_height
        return max(0.0, min(1.0, ratio * 8))

    def _compute_deviation(self, landmarks: List[Point2D]) -> float:
        chin = landmarks[self.CHIN]
        nose = landmarks[self.NOSE_TIP]
        forehead = landmarks[self.FOREHEAD_CENTER]

        face_center_x = (forehead.x + nose.x) / 2
        deviation = chin.x - face_center_x

        face_width = landmarks[self.JAW_LEFT].distance_to(
            landmarks[self.JAW_RIGHT]
        )
        if face_width < 1e-6:
            return 0.0

        normalized = deviation / (face_width / 2)
        return max(-1.0, min(1.0, normalized))

    def _compute_clench(self, landmarks: List[Point2D]) -> float:
        upper_lip = landmarks[self.UPPER_LIP]
        lower_lip = landmarks[self.LOWER_LIP]
        lip_distance = upper_lip.distance_to(lower_lip)

        face_height = landmarks[self.FOREHEAD_CENTER].distance_to(
            landmarks[self.CHIN]
        )
        if face_height < 1e-6:
            return 0.0

        ratio = lip_distance / face_height
        clench = max(0.0, min(1.0, 1.0 - ratio * 15))
        return clench
