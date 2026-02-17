from typing import List, Optional
import math

from spectral.models import Point2D, MouthData, TrackingConfig


class MouthTracker:
    UPPER_OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LOWER_OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    UPPER_INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    LOWER_INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

    LEFT_CORNER = 61
    RIGHT_CORNER = 291
    UPPER_LIP_CENTER = 13
    LOWER_LIP_CENTER = 14
    UPPER_LIP_TOP = 0
    LOWER_LIP_BOTTOM = 17

    ALL_MOUTH_INDICES = list(
        set(
            UPPER_OUTER_LIP
            + LOWER_OUTER_LIP
            + UPPER_INNER_LIP
            + LOWER_INNER_LIP
        )
    )

    NOSE_TIP = 1
    CHIN = 152
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454

    def __init__(self, config: TrackingConfig):
        self._config = config

    def track(self, landmarks: List[Point2D]) -> Optional[MouthData]:
        if len(landmarks) < 468:
            return None

        mouth_landmarks = [landmarks[i] for i in self.ALL_MOUTH_INDICES]
        cx = sum(p.x for p in mouth_landmarks) / len(mouth_landmarks)
        cy = sum(p.y for p in mouth_landmarks) / len(mouth_landmarks)
        center = Point2D(cx, cy)

        upper_lip_center = landmarks[self.UPPER_LIP_CENTER]
        lower_lip_center = landmarks[self.LOWER_LIP_CENTER]
        left_corner = landmarks[self.LEFT_CORNER]
        right_corner = landmarks[self.RIGHT_CORNER]

        width = left_corner.distance_to(right_corner)
        height = upper_lip_center.distance_to(lower_lip_center)

        outer_height = landmarks[self.UPPER_LIP_TOP].distance_to(
            landmarks[self.LOWER_LIP_BOTTOM]
        )

        if width < 1e-6:
            aspect_ratio = 0.0
        else:
            aspect_ratio = height / width

        openness = self._compute_openness(landmarks, height, width)
        is_open = openness > self._config.mouth_open_threshold

        smile_amount = self._compute_smile(landmarks)
        pucker_amount = self._compute_pucker(landmarks, width, outer_height)

        return MouthData(
            center=center,
            landmarks=mouth_landmarks,
            upper_lip_center=upper_lip_center,
            lower_lip_center=lower_lip_center,
            left_corner=left_corner,
            right_corner=right_corner,
            openness=openness,
            width=width,
            height=height,
            smile_amount=smile_amount,
            pucker_amount=pucker_amount,
            aspect_ratio=aspect_ratio,
            is_open=is_open,
        )

    def _compute_openness(
        self, landmarks: List[Point2D], inner_height: float, width: float
    ) -> float:
        if width < 1e-6:
            return 0.0
        ratio = inner_height / width
        return max(0.0, min(1.0, ratio * 2.5))

    def _compute_smile(self, landmarks: List[Point2D]) -> float:
        left_corner = landmarks[self.LEFT_CORNER]
        right_corner = landmarks[self.RIGHT_CORNER]

        nose_tip = landmarks[self.NOSE_TIP]
        corner_mid_y = (left_corner.y + right_corner.y) / 2

        mouth_center_y = landmarks[self.UPPER_LIP_CENTER].y
        vertical_diff = mouth_center_y - corner_mid_y

        face_height = landmarks[self.NOSE_TIP].distance_to(landmarks[self.CHIN])
        if face_height < 1e-6:
            return 0.0

        normalized = vertical_diff / face_height
        smile = max(0.0, min(1.0, normalized * 5 + 0.5))
        return smile

    def _compute_pucker(
        self, landmarks: List[Point2D], width: float, height: float
    ) -> float:
        if height < 1e-6:
            return 0.0
        ratio = width / height
        pucker = max(0.0, min(1.0, 1.0 - (ratio - 1.0) / 3.0))
        return pucker
