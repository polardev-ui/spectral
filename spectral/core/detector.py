import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple

from spectral.models import (
    Point2D,
    Point3D,
    BoundingBox,
    FaceData,
    TrackingConfig,
)


class FaceDetector:
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]

    def __init__(self, config: TrackingConfig):
        self._config = config
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=config.static_image_mode,
            max_num_faces=config.max_faces,
            refine_landmarks=config.refine_landmarks,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )
        self._frame_count = 0

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[Optional[object], int, int]:
        self._frame_count += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        return results, w, h

    def extract_landmarks(
        self, results: object, width: int, height: int
    ) -> List[Tuple[List[Point2D], List[Point3D], BoundingBox, float]]:
        faces = []
        if not results.multi_face_landmarks:
            return faces

        for face_landmarks in results.multi_face_landmarks:
            points_2d = []
            points_3d = []
            x_coords = []
            y_coords = []

            for lm in face_landmarks.landmark:
                px = lm.x * width
                py = lm.y * height
                pz = lm.z * width
                points_2d.append(Point2D(px, py))
                points_3d.append(Point3D(px, py, pz))
                x_coords.append(px)
                y_coords.append(py)

            x_min = int(min(x_coords))
            y_min = int(min(y_coords))
            x_max = int(max(x_coords))
            y_max = int(max(y_coords))
            margin_x = int((x_max - x_min) * 0.05)
            margin_y = int((y_max - y_min) * 0.05)

            bbox = BoundingBox(
                x=max(0, x_min - margin_x),
                y=max(0, y_min - margin_y),
                width=min(width, x_max - x_min + 2 * margin_x),
                height=min(height, y_max - y_min + 2 * margin_y),
            )

            visibility_sum = sum(
                lm.visibility
                for lm in face_landmarks.landmark
                if hasattr(lm, "visibility")
            )
            confidence = min(
                1.0, visibility_sum / max(1, len(face_landmarks.landmark)) + 0.5
            )

            faces.append((points_2d, points_3d, bbox, confidence))

        return faces

    def release(self):
        self._face_mesh.close()
