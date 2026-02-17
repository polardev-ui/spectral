from typing import List, Optional

from spectral.models import Point2D, EyebrowData, TrackingConfig


class EyebrowTracker:
    LEFT_EYEBROW_INDICES = [276, 283, 282, 295, 285]
    RIGHT_EYEBROW_INDICES = [46, 53, 52, 65, 55]

    LEFT_EYEBROW_UPPER = [283, 282, 295]
    LEFT_EYEBROW_LOWER = [276, 285]
    RIGHT_EYEBROW_UPPER = [53, 52, 65]
    RIGHT_EYEBROW_LOWER = [46, 55]

    LEFT_EYE_TOP = 159
    RIGHT_EYE_TOP = 386

    LEFT_EYEBROW_INNER = 285
    LEFT_EYEBROW_OUTER = 276
    RIGHT_EYEBROW_INNER = 55
    RIGHT_EYEBROW_OUTER = 46

    NOSE_BRIDGE = 6

    def __init__(self, config: TrackingConfig):
        self._config = config
        self._baseline_left = None
        self._baseline_right = None
        self._calibration_frames = 0
        self._calibration_sum_left = 0.0
        self._calibration_sum_right = 0.0
        self._calibration_target = 30

    def _compute_raise_amount(
        self,
        landmarks: List[Point2D],
        eyebrow_indices: List[int],
        eye_top_idx: int,
    ) -> float:
        eyebrow_y = sum(landmarks[i].y for i in eyebrow_indices) / len(eyebrow_indices)
        eye_y = landmarks[eye_top_idx].y
        distance = eye_y - eyebrow_y

        nose_bridge = landmarks[self.NOSE_BRIDGE]
        face_scale = abs(nose_bridge.y - landmarks[eye_top_idx].y)
        if face_scale < 1e-6:
            return 0.0

        normalized = distance / face_scale
        return max(0.0, min(1.0, normalized))

    def _compute_furrow(
        self,
        landmarks: List[Point2D],
        left_inner_idx: int,
        right_inner_idx: int,
    ) -> float:
        distance = landmarks[left_inner_idx].distance_to(landmarks[right_inner_idx])
        nose_width = landmarks[48].distance_to(landmarks[278])
        if nose_width < 1e-6:
            return 0.0
        ratio = distance / nose_width
        furrow = max(0.0, min(1.0, 1.0 - ratio))
        return furrow

    def _compute_arch_height(
        self,
        landmarks: List[Point2D],
        upper_indices: List[int],
        inner_idx: int,
        outer_idx: int,
    ) -> float:
        inner_point = landmarks[inner_idx]
        outer_point = landmarks[outer_idx]
        baseline_y = (inner_point.y + outer_point.y) / 2

        max_deviation = 0.0
        for idx in upper_indices:
            deviation = baseline_y - landmarks[idx].y
            max_deviation = max(max_deviation, deviation)

        span = inner_point.distance_to(outer_point)
        if span < 1e-6:
            return 0.0

        return max(0.0, min(1.0, max_deviation / (span * 0.3)))

    def track_left_eyebrow(self, landmarks: List[Point2D]) -> Optional[EyebrowData]:
        return self._track_eyebrow(
            landmarks,
            self.LEFT_EYEBROW_INDICES,
            self.LEFT_EYEBROW_UPPER,
            self.LEFT_EYEBROW_INNER,
            self.LEFT_EYEBROW_OUTER,
            self.LEFT_EYE_TOP,
        )

    def track_right_eyebrow(self, landmarks: List[Point2D]) -> Optional[EyebrowData]:
        return self._track_eyebrow(
            landmarks,
            self.RIGHT_EYEBROW_INDICES,
            self.RIGHT_EYEBROW_UPPER,
            self.RIGHT_EYEBROW_INNER,
            self.RIGHT_EYEBROW_OUTER,
            self.RIGHT_EYE_TOP,
        )

    def _track_eyebrow(
        self,
        landmarks: List[Point2D],
        indices: List[int],
        upper_indices: List[int],
        inner_idx: int,
        outer_idx: int,
        eye_top_idx: int,
    ) -> Optional[EyebrowData]:
        if len(landmarks) < max(indices) + 1:
            return None

        brow_landmarks = [landmarks[i] for i in indices]
        cx = sum(p.x for p in brow_landmarks) / len(brow_landmarks)
        cy = sum(p.y for p in brow_landmarks) / len(brow_landmarks)
        center = Point2D(cx, cy)

        raise_amount = self._compute_raise_amount(landmarks, indices, eye_top_idx)

        furrow_amount = self._compute_furrow(
            landmarks, self.LEFT_EYEBROW_INNER, self.RIGHT_EYEBROW_INNER
        )

        arch_height = self._compute_arch_height(
            landmarks, upper_indices, inner_idx, outer_idx
        )

        return EyebrowData(
            landmarks=brow_landmarks,
            center=center,
            raise_amount=raise_amount,
            furrow_amount=furrow_amount,
            arch_height=arch_height,
            inner_point=landmarks[inner_idx],
            outer_point=landmarks[outer_idx],
        )
