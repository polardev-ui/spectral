import sys
import os
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral import SpectralEngine, TrackingConfig
from spectral.models import GestureType


class PuppetFace:
    def __init__(self, width=500, height=500):
        self._width = width
        self._height = height
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)

        self._face_center = (width // 2, height // 2)
        self._face_radius = 150
        self._eye_radius = 25
        self._iris_radius = 10
        self._mouth_width = 80
        self._mouth_height = 15

        self._left_eye_open = 1.0
        self._right_eye_open = 1.0
        self._left_iris_x = 0.0
        self._left_iris_y = 0.0
        self._right_iris_x = 0.0
        self._right_iris_y = 0.0
        self._mouth_open = 0.0
        self._smile = 0.5
        self._left_brow = 0.0
        self._right_brow = 0.0
        self._head_roll = 0.0
        self._head_yaw = 0.0

    def update(self, face):
        if face.left_eye:
            self._left_eye_open = face.left_eye.openness
            if face.left_eye.gaze_direction:
                self._left_iris_x = face.left_eye.gaze_direction.x
                self._left_iris_y = face.left_eye.gaze_direction.y

        if face.right_eye:
            self._right_eye_open = face.right_eye.openness
            if face.right_eye.gaze_direction:
                self._right_iris_x = face.right_eye.gaze_direction.x
                self._right_iris_y = face.right_eye.gaze_direction.y

        if face.mouth:
            self._mouth_open = face.mouth.openness
            self._smile = face.mouth.smile_amount

        if face.left_eyebrow:
            self._left_brow = face.left_eyebrow.raise_amount
        if face.right_eyebrow:
            self._right_brow = face.right_eyebrow.raise_amount

        if face.head_pose:
            self._head_roll = face.head_pose.roll
            self._head_yaw = face.head_pose.yaw

    def render(self):
        self._canvas[:] = (35, 35, 45)
        cx, cy = self._face_center

        yaw_offset = int(self._head_yaw * 1.5)

        cv2.circle(
            self._canvas,
            (cx + yaw_offset, cy),
            self._face_radius,
            (80, 140, 200),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            self._canvas,
            (cx + yaw_offset, cy),
            self._face_radius,
            (60, 110, 170),
            3,
            cv2.LINE_AA,
        )

        left_eye_cx = cx - 55 + yaw_offset
        left_eye_cy = cy - 30
        right_eye_cx = cx + 55 + yaw_offset
        right_eye_cy = cy - 30

        brow_offset_l = int(self._left_brow * 25)
        brow_offset_r = int(self._right_brow * 25)

        cv2.line(
            self._canvas,
            (left_eye_cx - 30, left_eye_cy - 35 - brow_offset_l),
            (left_eye_cx + 30, left_eye_cy - 30 - brow_offset_l),
            (60, 110, 170),
            4,
            cv2.LINE_AA,
        )
        cv2.line(
            self._canvas,
            (right_eye_cx - 30, right_eye_cy - 30 - brow_offset_r),
            (right_eye_cx + 30, right_eye_cy - 35 - brow_offset_r),
            (60, 110, 170),
            4,
            cv2.LINE_AA,
        )

        left_eye_h = max(2, int(self._eye_radius * self._left_eye_open))
        right_eye_h = max(2, int(self._eye_radius * self._right_eye_open))

        cv2.ellipse(
            self._canvas,
            (left_eye_cx, left_eye_cy),
            (self._eye_radius, left_eye_h),
            0,
            0,
            360,
            (240, 240, 240),
            -1,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            self._canvas,
            (right_eye_cx, right_eye_cy),
            (self._eye_radius, right_eye_h),
            0,
            0,
            360,
            (240, 240, 240),
            -1,
            cv2.LINE_AA,
        )

        if self._left_eye_open > 0.2:
            iris_x = left_eye_cx + int(self._left_iris_x * 12)
            iris_y = left_eye_cy + int(self._left_iris_y * 8)
            cv2.circle(
                self._canvas,
                (iris_x, iris_y),
                self._iris_radius,
                (80, 60, 40),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                self._canvas,
                (iris_x, iris_y),
                5,
                (30, 20, 10),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                self._canvas,
                (iris_x - 3, iris_y - 3),
                2,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )

        if self._right_eye_open > 0.2:
            iris_x = right_eye_cx + int(self._right_iris_x * 12)
            iris_y = right_eye_cy + int(self._right_iris_y * 8)
            cv2.circle(
                self._canvas,
                (iris_x, iris_y),
                self._iris_radius,
                (80, 60, 40),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                self._canvas,
                (iris_x, iris_y),
                5,
                (30, 20, 10),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                self._canvas,
                (iris_x - 3, iris_y - 3),
                2,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )

        nose_cx = cx + yaw_offset
        nose_cy = cy + 15
        pts = np.array(
            [
                [nose_cx, nose_cy - 10],
                [nose_cx - 12, nose_cy + 10],
                [nose_cx + 12, nose_cy + 10],
            ],
            np.int32,
        )
        cv2.fillPoly(self._canvas, [pts], (70, 120, 180), cv2.LINE_AA)

        mouth_cx = cx + yaw_offset
        mouth_cy = cy + 60
        mouth_h = max(2, int(20 * self._mouth_open + 5))
        mouth_w = int(self._mouth_width * (0.8 + self._smile * 0.4))

        if self._mouth_open > 0.3:
            cv2.ellipse(
                self._canvas,
                (mouth_cx, mouth_cy),
                (mouth_w, mouth_h),
                0,
                0,
                360,
                (50, 50, 80),
                -1,
                cv2.LINE_AA,
            )
            cv2.ellipse(
                self._canvas,
                (mouth_cx, mouth_cy),
                (mouth_w, mouth_h),
                0,
                0,
                360,
                (150, 80, 80),
                2,
                cv2.LINE_AA,
            )
        else:
            curve = int((self._smile - 0.4) * 40)
            pts = np.array(
                [
                    [mouth_cx - mouth_w, mouth_cy],
                    [mouth_cx - mouth_w // 2, mouth_cy + curve],
                    [mouth_cx, mouth_cy + curve + 2],
                    [mouth_cx + mouth_w // 2, mouth_cy + curve],
                    [mouth_cx + mouth_w, mouth_cy],
                ],
                np.int32,
            )
            cv2.polylines(
                self._canvas,
                [pts],
                False,
                (150, 80, 80),
                3,
                cv2.LINE_AA,
            )

        return self._canvas


def main():
    config = TrackingConfig(
        max_faces=1,
        enable_expression_analysis=False,
        enable_gesture_detection=False,
        smoothing_factor=0.5,
    )

    engine = SpectralEngine(config)
    puppet = PuppetFace(500, 500)

    print("Spectral - Virtual Puppet Demo")
    print("Press 'q' or ESC to quit")
    print("")
    print("Your facial movements control the virtual puppet.")
    print("Try blinking, smiling, raising eyebrows, and looking around.")
    print("")

    try:
        for frame, tracking in engine.get_data_stream(camera_index=0):
            if tracking.has_faces:
                puppet.update(tracking.primary_face)

            puppet_display = puppet.render()

            camera_small = cv2.resize(frame, (160, 120))
            camera_overlay = engine.draw(camera_small, tracking)
            puppet_display[10:130, 330:490] = camera_overlay

            cv2.imshow("Spectral - Virtual Puppet", puppet_display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        engine.release()


if __name__ == "__main__":
    main()
