from typing import List, Optional

from spectral.models import Point2D, NoseData, TrackingConfig


class NoseTracker:
    NOSE_TIP = 1
    NOSE_BRIDGE_TOP = 6
    NOSE_BRIDGE_INDICES = [6, 197, 195, 5, 4]
    NOSE_BOTTOM_INDICES = [2, 98, 327]
    LEFT_NOSTRIL = 48
    RIGHT_NOSTRIL = 278
    NOSE_CONTOUR = [
        48, 115, 220, 45, 4, 275, 440, 344, 278,
    ]
    NOSE_TIP_REGION = [1, 2, 98, 327, 4]

    LEFT_ALA = 219
    RIGHT_ALA = 439

    def __init__(self, config: TrackingConfig):
        self._config = config

    def track(self, landmarks: List[Point2D]) -> Optional[NoseData]:
        if len(landmarks) < 468:
            return None

        tip = landmarks[self.NOSE_TIP]
        bridge = landmarks[self.NOSE_BRIDGE_TOP]

        nose_landmarks = [landmarks[i] for i in self.NOSE_CONTOUR]

        nostril_left = landmarks[self.LEFT_NOSTRIL]
        nostril_right = landmarks[self.RIGHT_NOSTRIL]

        width = nostril_left.distance_to(nostril_right)
        length = bridge.distance_to(tip)

        wrinkle_amount = self._compute_wrinkle(landmarks)

        return NoseData(
            tip=tip,
            bridge=bridge,
            landmarks=nose_landmarks,
            nostril_left=nostril_left,
            nostril_right=nostril_right,
            width=width,
            length=length,
            wrinkle_amount=wrinkle_amount,
        )

    def _compute_wrinkle(self, landmarks: List[Point2D]) -> float:
        bridge_points = [landmarks[i] for i in self.NOSE_BRIDGE_INDICES]
        if len(bridge_points) < 3:
            return 0.0

        total_distance = 0.0
        for i in range(len(bridge_points) - 1):
            total_distance += bridge_points[i].distance_to(bridge_points[i + 1])

        straight_distance = bridge_points[0].distance_to(bridge_points[-1])
        if straight_distance < 1e-6:
            return 0.0

        curvature = (total_distance / straight_distance) - 1.0
        return max(0.0, min(1.0, curvature * 10))
