from typing import List, Optional
import math

from spectral.models import Point2D, EyeData, TrackingConfig


class EyeTracker:
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    LEFT_EYE_UPPER = [159, 160, 161, 158]
    LEFT_EYE_LOWER = [145, 144, 163, 153]
    RIGHT_EYE_UPPER = [386, 385, 384, 387]
    RIGHT_EYE_LOWER = [374, 380, 381, 382]

    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    def __init__(self, config: TrackingConfig):
        self._config = config
        self._prev_left_ratio = 0.0
        self._prev_right_ratio = 0.0

    def _compute_aspect_ratio(
        self,
        landmarks: List[Point2D],
        upper_indices: List[int],
        lower_indices: List[int],
        inner_idx: int,
        outer_idx: int,
    ) -> float:
        vertical_distances = []
        for u, l in zip(upper_indices, lower_indices):
            dist = landmarks[u].distance_to(landmarks[l])
            vertical_distances.append(dist)

        horizontal_distance = landmarks[inner_idx].distance_to(landmarks[outer_idx])
        if horizontal_distance < 1e-6:
            return 0.0

        avg_vertical = sum(vertical_distances) / len(vertical_distances)
        return avg_vertical / horizontal_distance

    def _compute_iris_center(
        self, landmarks: List[Point2D], iris_indices: List[int]
    ) -> Optional[Point2D]:
        if len(landmarks) <= max(iris_indices):
            return None
        x = sum(landmarks[i].x for i in iris_indices) / len(iris_indices)
        y = sum(landmarks[i].y for i in iris_indices) / len(iris_indices)
        return Point2D(x, y)

    def _compute_iris_radius(
        self, landmarks: List[Point2D], iris_indices: List[int]
    ) -> float:
        if len(landmarks) <= max(iris_indices):
            return 0.0
        center = self._compute_iris_center(landmarks, iris_indices)
        if center is None:
            return 0.0
        radii = [center.distance_to(landmarks[i]) for i in iris_indices[1:]]
        return sum(radii) / len(radii) if radii else 0.0

    def _compute_gaze_direction(
        self,
        iris_center: Optional[Point2D],
        eye_inner: Point2D,
        eye_outer: Point2D,
        eye_upper: Point2D,
        eye_lower: Point2D,
    ) -> Optional[Point2D]:
        if iris_center is None:
            return None

        eye_center_x = (eye_inner.x + eye_outer.x) / 2
        eye_center_y = (eye_upper.y + eye_lower.y) / 2
        eye_width = abs(eye_outer.x - eye_inner.x)
        eye_height = abs(eye_lower.y - eye_upper.y)

        if eye_width < 1e-6 or eye_height < 1e-6:
            return Point2D(0.0, 0.0)

        gaze_x = (iris_center.x - eye_center_x) / (eye_width / 2)
        gaze_y = (iris_center.y - eye_center_y) / (eye_height / 2)

        gaze_x = max(-1.0, min(1.0, gaze_x))
        gaze_y = max(-1.0, min(1.0, gaze_y))

        return Point2D(gaze_x, gaze_y)

    def _compute_pupil_dilation(
        self, iris_radius: float, eye_width: float
    ) -> float:
        if eye_width < 1e-6:
            return 0.0
        return min(1.0, (iris_radius * 2) / eye_width)

    def track_left_eye(self, landmarks: List[Point2D]) -> Optional[EyeData]:
        return self._track_eye(
            landmarks,
            self.LEFT_EYE_INDICES,
            self.LEFT_EYE_UPPER,
            self.LEFT_EYE_LOWER,
            self.LEFT_IRIS_INDICES,
            self.LEFT_EYE_INNER,
            self.LEFT_EYE_OUTER,
        )

    def track_right_eye(self, landmarks: List[Point2D]) -> Optional[EyeData]:
        return self._track_eye(
            landmarks,
            self.RIGHT_EYE_INDICES,
            self.RIGHT_EYE_UPPER,
            self.RIGHT_EYE_LOWER,
            self.RIGHT_IRIS_INDICES,
            self.RIGHT_EYE_INNER,
            self.RIGHT_EYE_OUTER,
        )

    def _track_eye(
        self,
        landmarks: List[Point2D],
        eye_indices: List[int],
        upper_indices: List[int],
        lower_indices: List[int],
        iris_indices: List[int],
        inner_idx: int,
        outer_idx: int,
    ) -> Optional[EyeData]:
        if len(landmarks) < 468:
            return None

        eye_landmarks = [landmarks[i] for i in eye_indices]
        cx = sum(p.x for p in eye_landmarks) / len(eye_landmarks)
        cy = sum(p.y for p in eye_landmarks) / len(eye_landmarks)
        center = Point2D(cx, cy)

        aspect_ratio = self._compute_aspect_ratio(
            landmarks, upper_indices, lower_indices, inner_idx, outer_idx
        )

        iris_center = None
        iris_radius = 0.0
        if self._config.enable_iris_tracking:
            iris_center = self._compute_iris_center(landmarks, iris_indices)
            iris_radius = self._compute_iris_radius(landmarks, iris_indices)

        gaze = self._compute_gaze_direction(
            iris_center,
            landmarks[inner_idx],
            landmarks[outer_idx],
            landmarks[upper_indices[0]],
            landmarks[lower_indices[0]],
        )

        eye_width = landmarks[inner_idx].distance_to(landmarks[outer_idx])
        pupil_dilation = self._compute_pupil_dilation(iris_radius, eye_width)

        is_closed = aspect_ratio < self._config.blink_threshold
        openness = min(1.0, aspect_ratio / 0.4)

        return EyeData(
            center=center,
            landmarks=eye_landmarks,
            iris_center=iris_center,
            iris_radius=iris_radius,
            openness=openness,
            gaze_direction=gaze,
            pupil_dilation=pupil_dilation,
            aspect_ratio=aspect_ratio,
            is_closed=is_closed,
        )
