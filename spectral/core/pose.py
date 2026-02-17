import numpy as np
import cv2
from typing import List, Optional, Tuple

from spectral.models import Point2D, Point3D, HeadPose


class PoseEstimator:
    MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype=np.float64,
    )

    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(self):
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._prev_rvec = None
        self._prev_tvec = None

    def _get_camera_matrix(self, width: int, height: int) -> np.ndarray:
        if self._camera_matrix is None:
            focal_length = width
            center = (width / 2, height / 2)
            self._camera_matrix = np.array(
                [
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        return self._camera_matrix

    def estimate(
        self,
        landmarks_2d: List[Point2D],
        landmarks_3d: List[Point3D],
        frame_width: int,
        frame_height: int,
    ) -> Optional[HeadPose]:
        if len(landmarks_2d) < max(self.LANDMARK_INDICES) + 1:
            return None

        image_points = np.array(
            [landmarks_2d[i].as_tuple() for i in self.LANDMARK_INDICES],
            dtype=np.float64,
        )

        camera_matrix = self._get_camera_matrix(frame_width, frame_height)

        if self._prev_rvec is not None:
            success, rvec, tvec = cv2.solvePnP(
                self.MODEL_POINTS,
                image_points,
                camera_matrix,
                self._dist_coeffs,
                rvec=self._prev_rvec.copy(),
                tvec=self._prev_tvec.copy(),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            success, rvec, tvec = cv2.solvePnP(
                self.MODEL_POINTS,
                image_points,
                camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if not success:
            return None

        self._prev_rvec = rvec
        self._prev_tvec = tvec

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        projection_matrix = np.hstack((rotation_matrix, tvec))
        euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)[6]

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        if pitch > 180:
            pitch -= 360
        if yaw > 180:
            yaw -= 360
        if roll > 180:
            roll -= 360

        position = Point3D(
            float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])
        )

        rot_list = rotation_matrix.tolist()
        trans_list = [float(tvec[i][0]) for i in range(3)]

        return HeadPose(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            position=position,
            rotation_matrix=rot_list,
            translation_vector=trans_list,
        )

    def reset(self):
        self._prev_rvec = None
        self._prev_tvec = None
        self._camera_matrix = None
